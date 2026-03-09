"""
Autoresearch pretraining script. Single-GPU, single-file.
V100-compatible refactor with explicit support for standard, shared-block,
and recurrent / recursive-style transformer execution.
Usage: uv run train.py
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from kernels import get_kernel
except Exception:
    get_kernel = None

from prepare import MAX_SEQ_LEN, TIME_BUDGET, Tokenizer, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Platform / runtime config
# ---------------------------------------------------------------------------

USE_TORCH_COMPILE = False
USE_FLASH_ATTENTION_IF_AVAILABLE = True


def _compile(fn):
    if USE_TORCH_COMPILE:
        return torch.compile(fn, dynamic=False, fullgraph=True)
    return fn


def detect_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("autoresearch requires CUDA")
    cap = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name(0)
    major, minor = cap
    is_hopper = cap == (9, 0)
    # bfloat16 is not a good default on V100
    use_bf16 = major >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    peak_flops = 989.5e12 if use_bf16 else 125e12  # rough H100 bf16 / V100 fp16 proxies
    return {
        "capability": cap,
        "device_name": device_name,
        "is_hopper": is_hopper,
        "amp_dtype": amp_dtype,
        "peak_flops": peak_flops,
    }


RUNTIME = detect_runtime()
DEVICE = torch.device("cuda")
AUTocast_CTX = torch.amp.autocast(device_type="cuda", dtype=RUNTIME["amp_dtype"])


class AttentionBackend:
    def __init__(self):
        self.name = "sdpa"
        self.fa3 = None
        if USE_FLASH_ATTENTION_IF_AVAILABLE and get_kernel is not None and RUNTIME["is_hopper"]:
            try:
                repo = "varunneal/flash-attention-3"
                self.fa3 = get_kernel(repo).flash_attn_interface
                self.name = "flash-attn3"
            except Exception:
                self.fa3 = None
                self.name = "sdpa"

    @staticmethod
    def _build_additive_mask(T, device, window_size):
        mask = torch.zeros(T, T, device=device, dtype=torch.float32)
        upper = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
        mask = mask.masked_fill(upper, float("-inf"))
        if window_size is not None:
            win = window_size[0] if isinstance(window_size, tuple) else int(window_size)
            if win > 0 and win < T:
                idx = torch.arange(T, device=device)
                too_old = idx[None, :] < (idx[:, None] - (win - 1))
                mask = mask.masked_fill(too_old, float("-inf"))
        return mask.view(1, 1, T, T)

    def run(self, q, k, v, causal=True, window_size=None):
        # q/k/v are [B, T, H, D]
        if self.fa3 is not None:
            return self.fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if k.size(1) != q.size(1):
            repeat = q.size(1) // k.size(1)
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # Check if we actually need a window mask (short window < seq_len)
        need_window_mask = False
        if window_size is not None:
            win = window_size[0] if isinstance(window_size, tuple) else int(window_size)
            if 0 < win < q.size(-2):
                need_window_mask = True
        if causal and need_window_mask:
            attn_mask = self._build_additive_mask(q.size(-2), q.device, window_size)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        elif causal:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        return y.transpose(1, 2).contiguous()  # [B, T, H, D]


ATTN_BACKEND = AttentionBackend()

# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 8
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    window_pattern: str = "L"
    architecture_mode: str = "standard"  # standard | shared_block | recurrent
    num_unroll_steps: int = 8
    num_shared_blocks: int = 1
    use_depth_embedding: bool = True
    use_recurrent_state: bool = False
    state_dim: int = 128
    attention_every: int = 1
    mlp_ratio: int = 4


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], dim=-1)


class AttentionProj(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_head % self.n_kv_head == 0
        self.q_proj = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        return q, k, v


class AttentionCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, q, k, v, cos_sin, window_size):
        cos, sin = cos_sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = norm(q)
        k = norm(k)
        return ATTN_BACKEND.run(q, k, v, causal=True, window_size=window_size)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = AttentionProj(config)
        self.core = AttentionCore(config)
        self.o_proj = self.proj.o_proj

    def forward(self, x, cos_sin, window_size):
        B, T, _ = x.shape
        q, k, v = self.proj(x)
        y = self.core(q, k, v, cos_sin, window_size)
        y = y.view(B, T, -1)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.mlp_ratio * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        return self.c_proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, window_size, run_attention=True):
        if run_attention:
            x = x + self.attn(norm(x), cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class StateUpdate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.state_dim = config.state_dim
        self.in_proj = nn.Linear(config.n_embd + config.state_dim, config.state_dim, bias=False)
        self.gate_proj = nn.Linear(config.n_embd + config.state_dim, config.state_dim, bias=False)
        self.state_to_model = nn.Linear(config.state_dim, config.n_embd, bias=False)

    def init_state(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.state_dim, device=device, dtype=dtype)

    def inject(self, x, state):
        return x + self.state_to_model(state).unsqueeze(1)

    def update(self, x, state):
        pooled = x.mean(dim=1)
        fused = torch.cat([pooled, state], dim=-1)
        proposal = torch.tanh(self.in_proj(fused))
        gate = torch.sigmoid(self.gate_proj(fused))
        return gate * state + (1.0 - gate) * proposal


class StandardStack(nn.Module):
    def __init__(self, config, window_sizes):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.window_sizes = window_sizes
        self.effective_depth = config.n_layer
        self.unique_blocks = config.n_layer

    def forward(self, x, cos_sin, step_embed=None, state=None):
        for i, block in enumerate(self.blocks):
            x = block(x, cos_sin, self.window_sizes[i], run_attention=True)
        return x, state


class SharedBlockStack(nn.Module):
    def __init__(self, config, window_sizes):
        super().__init__()
        n_shared = max(1, config.num_shared_blocks)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n_shared)])
        self.window_sizes = window_sizes
        self.num_unroll_steps = config.num_unroll_steps
        self.use_depth_embedding = config.use_depth_embedding
        self.depth_embed = nn.Embedding(config.num_unroll_steps, config.n_embd) if self.use_depth_embedding else None
        self.attention_every = max(1, config.attention_every)
        self.effective_depth = config.num_unroll_steps
        self.unique_blocks = n_shared

    def forward(self, x, cos_sin, step_embed=None, state=None):
        for step in range(self.num_unroll_steps):
            block = self.blocks[step % len(self.blocks)]
            if self.depth_embed is not None:
                x = x + self.depth_embed.weight[step].view(1, 1, -1)
            run_attention = (step % self.attention_every) == 0 or step == self.num_unroll_steps - 1
            window = self.window_sizes[min(step, len(self.window_sizes) - 1)]
            x = block(x, cos_sin, window, run_attention=run_attention)
        return x, state


class RecurrentStack(nn.Module):
    def __init__(self, config, window_sizes):
        super().__init__()
        n_shared = max(1, config.num_shared_blocks)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(n_shared)])
        self.state_update = StateUpdate(config)
        self.window_sizes = window_sizes
        self.num_unroll_steps = config.num_unroll_steps
        self.use_depth_embedding = config.use_depth_embedding
        self.depth_embed = nn.Embedding(config.num_unroll_steps, config.n_embd) if self.use_depth_embedding else None
        self.attention_every = max(1, config.attention_every)
        self.effective_depth = config.num_unroll_steps
        self.unique_blocks = n_shared

    def forward(self, x, cos_sin, step_embed=None, state=None):
        if state is None:
            state = self.state_update.init_state(x.size(0), x.device, x.dtype)
        for step in range(self.num_unroll_steps):
            block = self.blocks[step % len(self.blocks)]
            if self.depth_embed is not None:
                x = x + self.depth_embed.weight[step].view(1, 1, -1)
            x = self.state_update.inject(x, state)
            run_attention = (step % self.attention_every) == 0 or step == self.num_unroll_steps - 1
            window = self.window_sizes[min(step, len(self.window_sizes) - 1)]
            x = block(x, cos_sin, window, run_attention=run_attention)
            state = self.state_update.update(x, state)
        return x, state


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.rotary_seq_len = config.sequence_len * 2
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.window_sizes = self._compute_window_sizes(config)
        self.stack = self._build_stack(config)

    def _build_stack(self, config):
        if config.architecture_mode == "standard":
            return StandardStack(config, self.window_sizes)
        if config.architecture_mode == "shared_block":
            return SharedBlockStack(config, self.window_sizes)
        if config.architecture_mode == "recurrent":
            return RecurrentStack(config, self.window_sizes)
        raise ValueError(f"unknown architecture_mode: {config.architecture_mode}")

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
        for module in self.modules():
            if isinstance(module, AttentionProj):
                torch.nn.init.uniform_(module.q_proj.weight, -s, s)
                torch.nn.init.uniform_(module.k_proj.weight, -s, s)
                torch.nn.init.uniform_(module.v_proj.weight, -s, s)
                torch.nn.init.zeros_(module.o_proj.weight)
            elif isinstance(module, MLP):
                torch.nn.init.uniform_(module.c_fc.weight, -s, s)
                torch.nn.init.zeros_(module.c_proj.weight)
            elif isinstance(module, StateUpdate):
                torch.nn.init.uniform_(module.in_proj.weight, -s, s)
                torch.nn.init.zeros_(module.gate_proj.weight)
                torch.nn.init.zeros_(module.state_to_model.weight)
            elif isinstance(module, nn.Embedding) and module is not self.transformer.wte:
                torch.nn.init.zeros_(module.weight)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=DEVICE)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=DEVICE)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos = cos.to(dtype=RUNTIME["amp_dtype"])[None, :, None, :]
        sin = sin.to(dtype=RUNTIME["amp_dtype"])[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        mapping = {"L": (long_window, 0), "S": (short_window, 0)}
        depth = max(config.n_layer, config.num_unroll_steps)
        window_sizes = [mapping[pattern[i % len(pattern)]] for i in range(depth)]
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        steps = getattr(self.stack, "effective_depth", self.config.n_layer)
        for i in range(steps):
            if self.config.architecture_mode != "standard" and self.config.attention_every > 1:
                run_attention = (i % self.config.attention_every) == 0 or i == steps - 1
            else:
                run_attention = True
            if not run_attention:
                continue
            window = self.window_sizes[min(i, len(self.window_sizes) - 1)][0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * nparams + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        stack = sum(p.numel() for p in self.stack.parameters())
        total = wte + lm_head + stack
        return {
            "wte": wte,
            "lm_head": lm_head,
            "stack": stack,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        matrix_params = []
        other_adamw_params = []
        for p in self.stack.parameters():
            if p.ndim == 2:
                matrix_params.append(p)
            else:
                other_adamw_params.append(p)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
        if other_adamw_params:
            param_groups.append(dict(kind='adamw', params=other_adamw_params, lr=scalar_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0))
        for shape in sorted({tuple(p.shape) for p in matrix_params}):
            group_params = [p for p in matrix_params if tuple(p.shape) == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, reduction='mean'):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x, _ = self.stack(x, cos_sin, state=None)
        x = norm(x)
        logits = self.lm_head(x).float()
        softcap = 15.0
        logits = softcap * torch.tanh(logits / softcap)
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=reduction)
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@_compile
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

@_compile
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.to(torch.float32)  # fp16 overflows in Newton-Schulz on V100; use fp32
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X.to(stacked_grads.dtype)
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'], self._adamw_step_t,
                             self._adamw_lr_t, self._adamw_beta1_t, self._adamw_beta2_t,
                             self._adamw_eps_t, self._adamw_wd_t)

    def _step_muon(self, group):
        params = group['params']
        if not params:
            return
        p0 = params[0]
        state = self.state[p0]
        num_params = len(params)
        shape, device, dtype = p0.shape, p0.device, p0.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack([p.data for p in params])
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        muon_step_fused(stacked_grads, stacked_params, state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        for p, new_p in zip(params, stacked_params.unbind(0)):
            p.data.copy_(new_p)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
ARCHITECTURE_MODE = "recurrent"   # standard | shared_block | recurrent
ASPECT_RATIO = 64                 # model_dim ~= depth * ASPECT_RATIO
HEAD_DIM = 64                     # smaller head dim is more V100-friendly
WINDOW_PATTERN = "L"             # V100-friendly default
NUM_UNROLL_STEPS = 10              # execution depth for shared / recurrent modes
NUM_SHARED_BLOCKS = 1             # unique blocks when sharing
USE_DEPTH_EMBEDDING = True
USE_RECURRENT_STATE = False
STATE_DIM = 128
ATTENTION_EVERY = 1
MLP_RATIO = 4

# Optimization
TOTAL_BATCH_SIZE = 2**15          # ~32K tokens / optimizer step, V100-friendly
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.1
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size
DEPTH = 8
DEVICE_BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_float32_matmul_precision("high")
print(f"Device: {RUNTIME['device_name']} | capability={RUNTIME['capability']} | backend={ATTN_BACKEND.name} | amp_dtype={RUNTIME['amp_dtype']} | compile={USE_TORCH_COMPILE}")

tokenizer = Tokenizer.from_directory()
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")


def build_model_config(depth):
    effective_depth = NUM_UNROLL_STEPS if ARCHITECTURE_MODE in {"shared_block", "recurrent"} else depth
    base_dim = max(depth, NUM_SHARED_BLOCKS) * ASPECT_RATIO
    model_dim = max(HEAD_DIM, ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM)
    num_heads = max(1, model_dim // HEAD_DIM)
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        architecture_mode=ARCHITECTURE_MODE,
        num_unroll_steps=effective_depth,
        num_shared_blocks=NUM_SHARED_BLOCKS,
        use_depth_embedding=USE_DEPTH_EMBEDDING,
        use_recurrent_state=USE_RECURRENT_STATE,
        state_dim=STATE_DIM,
        attention_every=ATTENTION_EVERY,
        mlp_ratio=MLP_RATIO,
    )


config = build_model_config(DEPTH)
print(f"Model config: {asdict(config)}")

model = GPT(config).to(DEVICE)
model.init_weights()

param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
num_params = param_counts['total']
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

if USE_TORCH_COMPILE:
    model = torch.compile(model, dynamic=False)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def get_muon_momentum(step):
    frac = min(step / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


def get_weight_decay(progress):
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0.0
total_training_time = 0.0
step = 0

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(grad_accum_steps):
        with AUTocast_CTX:
            loss = model(x, y)
        train_loss = loss.detach()
        (loss / grad_accum_steps).backward()
        x, y, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    optimizer.step()
    model.zero_grad(set_to_none=True)

    train_loss_f = train_loss.item()
    if not math.isfinite(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased = smooth_train_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / RUNTIME["peak_flops"]
    remaining = max(0.0, TIME_BUDGET - total_training_time)
    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()

    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()

total_tokens = step * TOTAL_BATCH_SIZE
model.eval()
with AUTocast_CTX:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

t_end = time.time()
startup_time = t_start_training - t_start
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * max(step - 10, 0) / max(total_training_time, 1e-6) / RUNTIME["peak_flops"]
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
compute_proxy = (num_params / 1e6) * (total_tokens / 1e6)
shared_ratio = getattr(model.stack, 'unique_blocks', config.n_layer) / max(getattr(model.stack, 'effective_depth', config.n_layer), 1)

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"depth:            {DEPTH}")
print(f"architecture:     {ARCHITECTURE_MODE}")
print(f"effective_depth:  {getattr(model.stack, 'effective_depth', config.n_layer)}")
print(f"unique_blocks:    {getattr(model.stack, 'unique_blocks', config.n_layer)}")
print(f"attention_every:  {ATTENTION_EVERY}")
print(f"compute_proxy:    {compute_proxy:.2f}")
print(f"shared_ratio:     {shared_ratio:.4f}")
