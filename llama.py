import dataclasses
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


_TRITON_AVAILABLE = triton is not None and torch.cuda.is_available()


if triton is not None:
    @triton.jit
    def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, n_cols, eps, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offsets, mask=mask, other=0.0)
        mean_square = tl.sum(x * x, axis=0) / n_cols
        y = x * tl.rsqrt(mean_square + eps) * w
        tl.store(y_ptr + row * n_cols + offsets, y, mask=mask)


    @triton.jit
    def _rope_kernel(
        x_ptr, sin_ptr, cos_ptr, y_ptr, n_cols, head_dim, n_heads, seq_len,
        BLOCK_SIZE: tl.constexpr
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        half = head_dim // 2
        mask = offsets < half
        position = (row // n_heads) % seq_len
        x0 = tl.load(x_ptr + row * n_cols + offsets, mask=mask, other=0.0)
        x1 = tl.load(x_ptr + row * n_cols + half + offsets, mask=mask, other=0.0)
        sin = tl.load(sin_ptr + position * half + offsets, mask=mask, other=0.0)
        cos = tl.load(cos_ptr + position * half + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + row * n_cols + offsets, x0 * cos - x1 * sin, mask=mask)
        tl.store(y_ptr + row * n_cols + half + offsets, x0 * sin + x1 * cos, mask=mask)


    @triton.jit
    def _attention_kernel(
        q_ptr, k_ptr, v_ptr, out_ptr,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        n_q, n_k, n_heads_q, n_heads_k, head_dim, scale,
        BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        q_pos = pid % n_q
        q_head = (pid // n_q) % n_heads_q
        batch = pid // (n_q * n_heads_q)
        kv_head = q_head // (n_heads_q // n_heads_k)
        d = tl.arange(0, BLOCK_D)
        q = tl.load(q_ptr + batch * stride_qb + q_head * stride_qh + q_pos * stride_qm + d * stride_qd, mask=d < head_dim, other=0.0)
        m_i = -float("inf")
        l_i = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for start in tl.range(0, n_k, BLOCK_N):
            positions = start + tl.arange(0, BLOCK_N)
            valid = (positions < n_k) & (positions <= q_pos)
            k = tl.load(k_ptr + batch * stride_kb + kv_head * stride_kh + positions[:, None] * stride_kn + d[None, :] * stride_kd, mask=valid[:, None] & (d[None, :] < head_dim), other=0.0)
            scores = tl.sum(q[None, :] * k, axis=1) * scale
            scores = tl.where(valid, scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
            p = tl.exp2((scores - m_new) * 1.4426950408889634)
            v = tl.load(v_ptr + batch * stride_vb + kv_head * stride_vh + positions[:, None] * stride_vn + d[None, :] * stride_vd, mask=valid[:, None] & (d[None, :] < head_dim), other=0.0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = m_new
        out = acc / l_i
        tl.store(out_ptr + pid * head_dim + d, out, mask=d < head_dim)


def _next_power_of_two(value):
    return 1 << (value - 1).bit_length()


def _can_use_triton(*tensors):
    return _TRITON_AVAILABLE and all(t.is_cuda and t.is_contiguous() for t in tensors)


@dataclasses.dataclass
class ModelConfig:
    head_dim: int

    hidden_size: int

    intermediate_size: int

    num_attention_heads: int

    num_hidden_layers: int

    num_key_value_heads: int

    rms_norm_eps: float

    rope_theta: float

    torch_dtype: str

    vocab_size: int


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))

        self.eps = eps

    def forward(self, input):
        if _can_use_triton(input) and input.shape[-1] <= 4096:
            input_2d = input.reshape(-1, input.shape[-1])
            output = torch.empty_like(input_2d)
            _rms_norm_kernel[(input_2d.shape[0],)](
                input_2d, self.weight, output, input_2d.shape[1], self.eps,
                BLOCK_SIZE=_next_power_of_two(input_2d.shape[1]),
            )
            return output.reshape_as(input)
        return (
            input
            * torch.rsqrt(input.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            * self.weight
        )


class MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()

        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)

        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

        self.silu = nn.SiLU()

    def forward(self, input):
        return self.down_proj(self.silu(self.gate_proj(input)) * self.up_proj(input))


def apply_rotary_position_embedding(input, sin_table, cos_table):
    if _can_use_triton(input, sin_table, cos_table):
        batch_size, seq_len, num_heads, head_dim = input.shape
        input_contiguous = input.contiguous()
        output = torch.empty_like(input_contiguous)
        _rope_kernel[(batch_size * seq_len * num_heads,)](
            input_contiguous, sin_table, cos_table, output,
            head_dim, head_dim, num_heads, seq_len,
            BLOCK_SIZE=_next_power_of_two(head_dim // 2),
        )
        return output

    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    input_0 = input[..., : input.shape[-1] // 2]
    input_1 = input[..., input.shape[-1] // 2 :]
    input_0_rotated = input_0 * cos_table - input_1 * sin_table
    input_1_rotated = input_0 * sin_table + input_1 * cos_table

    return torch.cat((input_0_rotated, input_1_rotated), dim=-1)


def apply_scaled_dot_product_attention(query, key, value):
    batch_size, num_heads_q, seq_len_q, emb_dim = query.shape
    _, num_heads_k, seq_len_k, _ = key.shape
    num_heads_v = value.shape[1]

    if (
        _can_use_triton(query, key, value)
        and num_heads_q % num_heads_k == 0
        and num_heads_q % num_heads_v == 0
        and emb_dim <= 256
    ):
        output = torch.empty_like(query)
        _attention_kernel[
            (batch_size * num_heads_q * seq_len_q,)
        ](
            query, key, value, output,
            *query.stride(), *key.stride(), *value.stride(),
            seq_len_q, seq_len_k, num_heads_q, num_heads_k, emb_dim,
            1 / math.sqrt(emb_dim),
            BLOCK_N=64,
            BLOCK_D=_next_power_of_two(emb_dim),
        )
        return output

    key = key.repeat_interleave(num_heads_q // num_heads_k, 1)
    value = value.repeat_interleave(num_heads_q // num_heads_v, 1)

    scale = 1 / math.sqrt(emb_dim)
    attn_mask = torch.tril(
        torch.full((seq_len_q, seq_len_k), True, device=query.device)
    )

    attn_output = torch.matmul(query, key.permute(0, 1, 3, 2)) * scale
    attn_output = torch.where(attn_mask, attn_output, float("-inf"))
    attn_output = torch.softmax(attn_output, dim=-1)
    return torch.matmul(attn_output, value)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_attention_heads = config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_attention_heads * self.head_dim, bias=False
        )

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states, sin_table, cos_table):
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape).permute(0, 2, 1, 3)

        query_states = apply_rotary_position_embedding(
            query_states, sin_table, cos_table
        ).permute(0, 2, 1, 3).contiguous()
        key_states = apply_rotary_position_embedding(
            key_states, sin_table, cos_table
        ).permute(0, 2, 1, 3).contiguous()
        value_states = value_states.contiguous()

        attn_output = apply_scaled_dot_product_attention(
            query_states, key_states, value_states
        )

        return self.o_proj(
            attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        )


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.self_attn = Attention(config)

        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states, sin_table, cos_table):
        hidden_states += self.self_attn(
            self.input_layernorm(hidden_states), sin_table, cos_table
        )

        hidden_states += self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states


def generate_sin_and_cos_tables(seq_len, emb_dim, base, dtype, device):
    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )

    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta)
    cos_table = torch.cos(positions * theta)

    return sin_table, cos_table


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.head_dim = config.head_dim

        self.hidden_size = config.hidden_size

        self.num_hidden_layers = config.num_hidden_layers

        self.rms_norm_eps = config.rms_norm_eps

        self.rope_theta = config.rope_theta

        self.torch_dtype = config.torch_dtype

        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(self.vocab_size, self.hidden_size)

        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(self.num_hidden_layers)
        )

        self.norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)

        seq_len = hidden_states.shape[1]

        sin_table, cos_table = generate_sin_and_cos_tables(
            seq_len,
            self.head_dim,
            base=self.rope_theta,
            dtype=getattr(torch, self.torch_dtype),
            device=input_ids.device,
        )

        for i in range(self.num_hidden_layers):
            hidden_states = self.layers[i](hidden_states, sin_table, cos_table)

        return self.norm(hidden_states)


class ModelForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = Model(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def generate(self, input_ids, max_new_tokens=20):
        for _ in range(max_new_tokens):
            hidden_states = self.model(input_ids)

            logits = self.lm_head(hidden_states[:, -1, :])

            next = torch.argmax(logits, dim=-1).unsqueeze(-1)

            input_ids = torch.cat((input_ids, next), dim=-1)

        return input_ids

    @staticmethod
    def from_pretrained(model_path):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            config = json.load(f)

        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]

        config = ModelConfig(
            **{
                key: value
                for key, value in config.items()
                if key in ModelConfig.__annotations__
            }
        )

        model = ModelForCausalLM(config).to(getattr(torch, config.torch_dtype))

        state_dict = load_file(model_path / "model.safetensors")

        if "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

        model.load_state_dict(state_dict)

        return model
