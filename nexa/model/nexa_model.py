"""NexaModel - Main transformer language model."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from nexa.model.components import RMSNorm, KVCache, precompute_rope_freqs
from nexa.model.attention import TransformerBlock


class NexaModel(nn.Module):
    def __init__(self, config, use_grad_ckpt=False):
        super().__init__()
        self.config = config
        self.use_grad_ckpt = use_grad_ckpt
        head_dim = config.n_embd // config.n_head

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f=RMSNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.critic_adapter = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.SiLU(),
            nn.Linear(config.n_embd, config.n_embd),
        )
        self.critic_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.critic_score_head = nn.Linear(config.n_embd, 1)
        self.memory_gate = nn.Linear(config.n_embd, config.n_embd)
        self.memory_query_gate = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.memory_value_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.memory_scale_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        freqs_cos, freqs_sin = precompute_rope_freqs(head_dim, config.block_size * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight") or pn.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        n_embd_params = config.vocab_size * config.n_embd
        n_non_embd = n_params - n_embd_params
        print(f"Model params : {n_params:,} ({n_params / 1e6:.1f}M)")
        print(f"  Embedding  : {n_embd_params:,} ({n_embd_params / 1e6:.1f}M) [vocab={config.vocab_size}]")
        print(f"  Transformer: {n_non_embd:,} ({n_non_embd / 1e6:.1f}M)")

        self.transformer.wte.weight.requires_grad_(True)
        self._current_entropy = None
        self._current_entropy_norm = None
        self._entropy_ema = None
        self._entropy_var_ema = None
        self._reflect_cooldown = 0
        self._spec_accept_ema = None
        self._adaptive_gamma = None
        self._spec_disable_steps = 0

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_memory_state(self, x, memory_state=None, memory_query_state=None):
        if memory_state is None:
            return x
        scale = getattr(self.config, "memory_state_scale", 0.08)
        if scale == 0:
            return x
        if memory_state.dim() == 1:
            memory_state = memory_state.unsqueeze(0)
        memory_state = memory_state.clamp(-3.0, 3.0).to(device=x.device, dtype=x.dtype)
        memory_value = self.memory_value_proj(memory_state).unsqueeze(1) * scale
        memory_scale = torch.tanh(self.memory_scale_proj(memory_state)).unsqueeze(1) * scale
        gate_logits = self.memory_gate(x)
        if memory_query_state is not None:
            if memory_query_state.dim() == 1:
                memory_query_state = memory_query_state.unsqueeze(0)
            memory_query_state = memory_query_state.to(device=x.device, dtype=x.dtype)
            if torch.norm(memory_query_state.float(), dim=-1).mean().item() >= 0.25:
                gate_logits = gate_logits + self.memory_query_gate(memory_query_state).unsqueeze(1)
        gate = torch.sigmoid(gate_logits)
        return x * (1.0 + gate * memory_scale) + gate * memory_value

    def _project_logits(self, x, head="main"):
        if head == "critic":
            return self.critic_head(x.detach())
        return self.lm_head(x)

    def forward(self, idx, targets=None, reasoning_targets=None, critic_labels=None,
                memory_state=None, memory_query_state=None, return_aux=False):
        B, T = idx.size()
        assert T <= self.freqs_cos.size(0), f"Sequence length {T} > max block_size {self.freqs_cos.size(0)}"

        x = self.transformer.drop(self.transformer.wte(idx))
        x = self._apply_memory_state(x, memory_state, memory_query_state)
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]
        for block in self.transformer.h:
            if self.use_grad_ckpt and self.training:
                x = block.forward_ckpt(x, freqs_cos, freqs_sin)
            else:
                x, _ = block(x, freqs_cos, freqs_sin, None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        need_critic = reasoning_targets is not None or critic_labels is not None
        critic_logits = None
        critic_score = None
        aux = {}

        if need_critic:
            critic_x = self.critic_adapter(x.detach())
            critic_logits = self.critic_head(critic_x)
            critic_score = torch.sigmoid(self.critic_score_head(critic_x[:, -1, :]))
            aux["critic_score"] = critic_score.detach()

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            loss = lm_loss
            aux["lm_loss"] = lm_loss.detach()
            if reasoning_targets is not None:
                reasoning_loss = F.cross_entropy(
                    critic_logits.view(-1, critic_logits.size(-1)),
                    reasoning_targets.view(-1), ignore_index=-100,
                )
                loss = loss + getattr(self.config, "reasoning_loss_weight", 0.1) * reasoning_loss
                aux["reasoning_loss"] = reasoning_loss.detach()
        if critic_labels is not None:
            critic_score_loss = F.binary_cross_entropy(
                critic_score, critic_labels.to(device=critic_score.device, dtype=critic_score.dtype),
            )
            loss = (critic_score_loss * getattr(self.config, "critic_score_loss_weight", 0.03)
                    if loss is None
                    else loss + getattr(self.config, "critic_score_loss_weight", 0.03) * critic_score_loss)
            aux["critic_score_loss"] = critic_score_loss.detach()
        if return_aux:
            return logits, loss, aux
        return logits, loss

    def _update_entropy_stats(self, logits):
        with torch.no_grad():
            _lp = F.log_softmax(logits.float(), dim=-1)
            ent = -(_lp.exp() * _lp).sum(dim=-1).mean()
        e = float(ent.item())
        e_norm = e / max(math.log(logits.size(-1)), 1e-8)
        self._current_entropy = e
        self._current_entropy_norm = e_norm
        if self._entropy_ema is None:
            self._entropy_ema = e_norm
            self._entropy_var_ema = 0.0
        else:
            beta = 0.95
            delta = e_norm - self._entropy_ema
            self._entropy_ema = beta * self._entropy_ema + (1 - beta) * e_norm
            self._entropy_var_ema = beta * self._entropy_var_ema + (1 - beta) * (delta * delta)
        return e

    def _sample_token(self, logits, generated, temperature, top_k, top_p, min_p,
                      repetition_penalty, seen_mask=None, track_entropy=True):
        """Sample with top-k, top-p, min-p, and repetition penalty."""
        if repetition_penalty != 1.0:
            vocab_size = logits.size(-1)
            for b in range(logits.size(0)):
                if seen_mask is not None:
                    score = logits[b].clone()
                    logits[b] = torch.where(
                        seen_mask,
                        torch.where(score > 0, score / repetition_penalty, score * repetition_penalty),
                        score,
                    )
                else:
                    seen = generated[b]
                    seen = seen[seen > 0]
                    if seen.numel() > 0:
                        seen = seen.clamp(0, vocab_size - 1)
                        score = logits[b].clone()
                        logits[b, seen] = torch.where(
                            score[seen] > 0, score[seen] / repetition_penalty, score[seen] * repetition_penalty,
                        )

        temperature = max(float(temperature), 1e-5)
        logits = logits / temperature
        raw_logits = logits.clone()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        if min_p > 0.0:
            probs = F.softmax(logits, dim=-1)
            max_probs = probs.max(dim=-1, keepdim=True).values
            mask = probs < (min_p * max_probs)
            logits[mask] = float("-inf")

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            sorted_logits[sorted_mask] = float("-inf")
            logits = torch.empty_like(logits).fill_(float("-inf")).scatter_(1, sorted_indices, sorted_logits)

        if track_entropy:
            self._update_entropy_stats(raw_logits)

        probs = F.softmax(logits, dim=-1)
        prob_sum = probs.sum(dim=-1, keepdim=True)

        if (prob_sum == 0).any():
            valid_mask = logits > float("-inf")
            probs = torch.where(valid_mask, 1.0, 0.0)
            prob_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        probs = probs / prob_sum.clamp(min=1e-8)
        return torch.multinomial(probs, num_samples=1)

    def _should_reflect(self, temperature: float) -> bool:
        if self._current_entropy is None:
            return False
        norm = self._current_entropy_norm or 0.0
        std = math.sqrt(max(self._entropy_var_ema or 0.0, 1e-8))
        z = (norm - (self._entropy_ema or norm)) / std
        z_th = 1.0 + 0.3 * max(0.0, temperature - 0.7)
        norm_th = 0.25 + 0.1 * max(0.0, temperature)
        return norm > norm_th and z > z_th and (getattr(self, "_reflect_cooldown", 0) <= 0)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=0.9,
                 min_p=0.05, repetition_penalty=1.1, caches=None, draft_model=None,
                 gamma=4, head="main", memory_state=None, memory_query_state=None):
        was_training = self.training
        self.eval()

        B, T = idx.size()

        if T == 0:
            raise ValueError("Cannot generate from empty input (T=0)")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
        if max_new_tokens >= self.config.block_size:
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) must be smaller than block_size ({self.config.block_size})"
            )

        original_draft_training = None
        if draft_model is not None:
            original_draft_training = draft_model.training
            draft_model.eval()

        try:
            if getattr(self, "_spec_disable_steps", 0) > 0:
                draft_model = None
                self._spec_disable_steps = max(0, self._spec_disable_steps - max_new_tokens)
            if hasattr(self, "_reflect_cooldown"):
                self._reflect_cooldown = max(0, self._reflect_cooldown - 1)
            n_layers = len(self.transformer.h)

            if T + max_new_tokens > self.config.block_size:
                idx = idx[:, -(self.config.block_size - max_new_tokens):]
                T = idx.size(1)

            if caches is None:
                max_cache_len = min(
                    T + max_new_tokens,
                    getattr(self.config, "sliding_window", None) or self.config.block_size,
                )
                head_dim = self.config.n_embd // self.config.n_head
                caches = [
                    KVCache(
                        B,
                        max_cache_len,
                        self.config.n_kv_head,
                        head_dim,
                        idx.device,
                        self.transformer.wte.weight.dtype,
                        n_global_tokens=getattr(self.config, "n_global_tokens", 0),
                    )
                    for _ in range(n_layers)
                ]

            x = self.transformer.drop(self.transformer.wte(idx))
            x = self._apply_memory_state(x, memory_state, memory_query_state)
            freqs_cos = self.freqs_cos[:T]
            freqs_sin = self.freqs_sin[:T]
            for i, block in enumerate(self.transformer.h):
                x, caches[i] = block(x, freqs_cos, freqs_sin, kv_cache=caches[i])
            x = self.transformer.ln_f(x)
            logits = self._project_logits(x[:, -1, :], head=head)

            max_total_len = T + max_new_tokens
            all_so_far = torch.zeros((B, max_total_len), dtype=torch.long, device=idx.device)
            all_so_far[:, :T] = idx
            current_len = T
            seen_mask = None
            if B == 1:
                seen_mask = torch.zeros(self.config.vocab_size, dtype=torch.bool, device=idx.device)
                init_ids = idx[0]
                init_ids = init_ids[(init_ids > 0) & (init_ids < self.config.vocab_size)]
                if init_ids.numel() > 0:
                    seen_mask[init_ids.unique()] = True

            idx_next = self._sample_token(
                logits,
                all_so_far[:, :current_len],
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                seen_mask=seen_mask,
            )
            all_so_far[:, current_len] = idx_next.squeeze()
            if seen_mask is not None:
                seen_mask[idx_next.squeeze().clamp(0, self.config.vocab_size - 1)] = True
            current_len += 1

            eos_id = getattr(self.config, "eos_id", None)
            eos_id = -1 if eos_id is None else int(eos_id)
            while current_len < T + max_new_tokens:
                if eos_id >= 0 and (idx_next.squeeze(-1) == eos_id).all():
                    break
                x = self.transformer.wte(idx_next)
                x = self._apply_memory_state(x, memory_state, memory_query_state)
                pos = (current_len - 1) % self.freqs_cos.size(0)
                fc = self.freqs_cos[pos: pos + 1]
                fs = self.freqs_sin[pos: pos + 1]
                for i, block in enumerate(self.transformer.h):
                    x, caches[i] = block(x, fc, fs, kv_cache=caches[i])
                x = self.transformer.ln_f(x)
                logits = self._project_logits(x[:, -1, :], head=head)
                idx_next = self._sample_token(
                    logits,
                    all_so_far[:, :current_len],
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    seen_mask=seen_mask,
                )
                all_so_far[:, current_len] = idx_next.squeeze()
                if seen_mask is not None:
                    seen_mask[idx_next.squeeze().clamp(0, self.config.vocab_size - 1)] = True
                current_len += 1

            return all_so_far[:, :current_len]
        finally:
            self.train(was_training)
            if draft_model is not None and original_draft_training is not None:
                draft_model.train(original_draft_training)

    @torch.no_grad()
    def generate_stream(self, idx, max_new_tokens, temperature=1.0, top_k=0, top_p=0.9,
                        min_p=0.05, repetition_penalty=1.1, head="main",
                        memory_state=None, memory_query_state=None):
        was_training = self.training
        current_len = 0
        orig_T = idx.size(1)
        try:
            self.eval()
            B, T = idx.size()
            orig_T = T
            if T == 0:
                raise ValueError("Cannot generate from empty input (T=0)")
            if max_new_tokens <= 0:
                raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
            if max_new_tokens >= self.config.block_size:
                raise ValueError(
                    f"max_new_tokens ({max_new_tokens}) must be smaller than block_size ({self.config.block_size})"
                )

            if B != 1:
                out = self.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    head=head,
                    memory_state=memory_state,
                    memory_query_state=memory_query_state,
                )
                for tok in out[0, T:].tolist():
                    yield tok
                return

            n_layers = len(self.transformer.h)
            if T + max_new_tokens > self.config.block_size:
                idx = idx[:, -(self.config.block_size - max_new_tokens):]
                T = idx.size(1)

            max_cache_len = min(
                T + max_new_tokens,
                getattr(self.config, "sliding_window", None) or self.config.block_size,
            )
            head_dim = self.config.n_embd // self.config.n_head
            caches = [
                KVCache(
                    B,
                    max_cache_len,
                    self.config.n_kv_head,
                    head_dim,
                    idx.device,
                    self.transformer.wte.weight.dtype,
                    n_global_tokens=getattr(self.config, "n_global_tokens", 0),
                )
                for _ in range(n_layers)
            ]

            x = self.transformer.drop(self.transformer.wte(idx))
            x = self._apply_memory_state(x, memory_state, memory_query_state)
            freqs_cos = self.freqs_cos[:T]
            freqs_sin = self.freqs_sin[:T]
            for i, block in enumerate(self.transformer.h):
                x, caches[i] = block(x, freqs_cos, freqs_sin, kv_cache=caches[i])
            x = self.transformer.ln_f(x)
            logits = self._project_logits(x[:, -1, :], head=head)

            all_so_far = idx.clone()
            seen_mask = torch.zeros(self.config.vocab_size, dtype=torch.bool, device=idx.device)
            init_ids = idx[0]
            init_ids = init_ids[(init_ids > 0) & (init_ids < self.config.vocab_size)]
            if init_ids.numel() > 0:
                seen_mask[init_ids.unique()] = True

            eos_id = getattr(self.config, "eos_id", None)
            eos_id = -1 if eos_id is None else int(eos_id)
            current_len = T
            for _ in range(max_new_tokens):
                idx_next = self._sample_token(
                    logits,
                    all_so_far,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    repetition_penalty,
                    seen_mask=seen_mask,
                )
                token_id = int(idx_next.item())
                yield token_id
                if 0 <= token_id < self.config.vocab_size:
                    seen_mask[token_id] = True
                if eos_id >= 0 and token_id == eos_id:
                    break
                all_so_far = torch.cat([all_so_far, idx_next], dim=1)
                x = self.transformer.wte(idx_next)
                x = self._apply_memory_state(x, memory_state, memory_query_state)
                pos = current_len % self.freqs_cos.size(0)
                fc = self.freqs_cos[pos: pos + 1]
                fs = self.freqs_sin[pos: pos + 1]
                for i, block in enumerate(self.transformer.h):
                    x, caches[i] = block(x, fc, fs, kv_cache=caches[i])
                x = self.transformer.ln_f(x)
                logits = self._project_logits(x[:, -1, :], head=head)
                current_len += 1
        except Exception as e:
            import logging
            logging.error(f"Generation stream failed at token {current_len - orig_T}: {e}")
            raise
        finally:
            self.train(was_training)
