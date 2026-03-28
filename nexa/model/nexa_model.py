"""NexaModel - main transformer language model for Nexa."""
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
        self.transformer.wte.weight = self.lm_head.weight

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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _project_logits(self, x):
        return self.lm_head(x)

    def forward(self, idx, targets=None, return_aux=False):
        _, T = idx.size()
        assert T <= self.freqs_cos.size(0), f"Sequence length {T} > max block_size {self.freqs_cos.size(0)}"

        x = self.transformer.drop(self.transformer.wte(idx))
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]
        for block in self.transformer.h:
            if self.use_grad_ckpt and self.training:
                x = block.forward_ckpt(x, freqs_cos, freqs_sin)
            else:
                x, _ = block(x, freqs_cos, freqs_sin, None)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        aux = {}

        loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            loss = lm_loss
            aux["lm_loss"] = lm_loss.detach()
        if return_aux:
            return logits, loss, aux
        return logits, loss

    def _sample_token(self, logits, generated, temperature, top_k, top_p, min_p,
                      repetition_penalty, seen_mask=None):
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

        probs = F.softmax(logits, dim=-1)
        prob_sum = probs.sum(dim=-1, keepdim=True)

        if (prob_sum == 0).any():
            valid_mask = logits > float("-inf")
            probs = torch.where(valid_mask, 1.0, 0.0)
            prob_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        probs = probs / prob_sum.clamp(min=1e-8)
        return torch.multinomial(probs, num_samples=1)

    def _prepare_generation_inputs(self, inputs, tokenizer=None):
        device = self.transformer.wte.weight.device
        if isinstance(inputs, torch.Tensor):
            idx = inputs.long().to(device)
            if idx.dim() == 1:
                idx = idx.unsqueeze(0)
            elif idx.dim() != 2:
                raise ValueError(f"inputs tensor must have shape (T,) or (B, T), got {tuple(idx.shape)}")
            prompt_lengths = self._infer_prompt_lengths(idx)
            return idx, prompt_lengths

        if isinstance(inputs, str):
            if tokenizer is None:
                raise ValueError("tokenizer is required when inputs are strings")
            rows = [tokenizer.encode(inputs).ids]
        elif isinstance(inputs, (list, tuple)):
            if not inputs:
                raise ValueError("inputs must not be empty")
            if all(isinstance(item, str) for item in inputs):
                if tokenizer is None:
                    raise ValueError("tokenizer is required when inputs contain strings")
                rows = [tokenizer.encode(item).ids for item in inputs]
            elif all(isinstance(item, int) for item in inputs):
                rows = [[int(item) for item in inputs]]
            else:
                rows = []
                for item in inputs:
                    if isinstance(item, torch.Tensor):
                        seq = item.tolist()
                    else:
                        seq = list(item)
                    rows.append([int(tok) for tok in seq])
        else:
            raise TypeError(
                "inputs must be a tensor, a string, a list of token ids, or a batch of those sequences"
            )

        if not rows or any(len(row) == 0 for row in rows):
            raise ValueError("Cannot generate from empty input")

        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.config, "eos_id", None)
        if pad_id is None:
            pad_id = 0
        max_len = max(len(row) for row in rows)
        idx = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
        prompt_lengths = []
        for batch_idx, row in enumerate(rows):
            prompt_lengths.append(len(row))
            idx[batch_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        return idx, prompt_lengths

    def _infer_prompt_lengths(self, idx):
        if idx.dim() != 2:
            raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")
        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None:
            return [idx.size(1)] * idx.size(0)

        prompt_lengths = []
        for row in idx:
            pad_positions = (row == int(pad_id)).nonzero(as_tuple=True)[0]
            length = int(pad_positions[0].item()) if pad_positions.numel() > 0 else row.size(0)
            if length <= 0:
                raise ValueError("Cannot generate from empty or fully padded input row")
            prompt_lengths.append(length)
        return prompt_lengths

    def _pad_token_rows(self, rows):
        device = self.transformer.wte.weight.device
        pad_id = getattr(self.config, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.config, "eos_id", None)
        if pad_id is None:
            pad_id = 0

        max_len = max((len(row) for row in rows), default=0)
        padded = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=device)
        for row_idx, row in enumerate(rows):
            if not row:
                continue
            padded[row_idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        return padded

    def _merge_generation_outputs(self, sample_outputs, tokenizer=None, return_dict=False, include_prompt=True):
        token_rows = []
        generated_rows = []
        prompt_lengths = []

        for output in sample_outputs:
            full_row = output.get("token_id_rows", [output["token_ids"][0].tolist()])[0]
            generated_row = output["generated_token_ids"][0]
            token_rows.append(full_row)
            generated_rows.append(generated_row)
            prompt_lengths.append(int(output["prompt_lengths"][0]))

        padded = self._pad_token_rows(token_rows)
        if not return_dict:
            return padded

        result = {
            "token_ids": padded,
            "generated_token_ids": generated_rows,
            "prompt_lengths": prompt_lengths,
        }
        if include_prompt:
            result["token_id_rows"] = token_rows
        if tokenizer is not None:
            result["generated_texts"] = tokenizer.decode_batch(generated_rows)
            if include_prompt:
                result["texts"] = tokenizer.decode_batch(token_rows)
        return result

    def _build_generation_output(self, full_sequence, prompt_lengths, tokenizer=None, return_dict=False, include_prompt=True):
        if not return_dict:
            return full_sequence

        full_rows = [full_sequence[i].tolist() for i in range(full_sequence.size(0))]
        generated_rows = [row[prompt_lengths[i] :] for i, row in enumerate(full_rows)]
        result = {
            "token_ids": full_sequence,
            "generated_token_ids": generated_rows,
            "prompt_lengths": prompt_lengths,
        }
        if include_prompt:
            result["token_id_rows"] = full_rows
        if tokenizer is not None:
            if hasattr(tokenizer, "decode_batch"):
                result["generated_texts"] = tokenizer.decode_batch(generated_rows)
                if include_prompt:
                    result["texts"] = tokenizer.decode_batch(full_rows)
            else:
                result["generated_texts"] = [tokenizer.decode(row) for row in generated_rows]
                if include_prompt:
                    result["texts"] = [tokenizer.decode(row) for row in full_rows]
        return result

    @torch.no_grad()
    def _generate_speculative_naive(
        self,
        idx,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty,
        draft_model,
        gamma,
        tokenizer=None,
        return_dict=False,
        include_prompt=True,
    ):
        current = idx.clone()
        eos_id = getattr(self.config, "eos_id", None)
        remaining = max_new_tokens

        while remaining > 0:
            step_gamma = min(int(gamma), remaining)
            draft_out = draft_model.generate(
                current,
                max_new_tokens=step_gamma,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )
            draft_tokens = draft_out[:, current.size(1) :]
            if draft_tokens.numel() == 0:
                break

            for draft_token in draft_tokens[0].tolist():
                target_out = self.generate(
                    current,
                    max_new_tokens=1,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    draft_model=None,
                )
                next_token = target_out[:, current.size(1) : current.size(1) + 1]
                current = torch.cat([current, next_token], dim=1)
                remaining -= 1
                if int(next_token.item()) != int(draft_token):
                    break
                if eos_id is not None and int(next_token.item()) == int(eos_id):
                    remaining = 0
                    break
                if remaining <= 0:
                    break

            if eos_id is not None and int(current[0, -1].item()) == int(eos_id):
                break

        return self._build_generation_output(
            current,
            [idx.size(1)] * idx.size(0),
            tokenizer=tokenizer,
            return_dict=return_dict,
            include_prompt=include_prompt,
        )

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        caches=None,
        draft_model=None,
        gamma=None,
        tokenizer=None,
        return_dict=False,
        include_prompt=True,
        use_speculative=None,
        eos_id=None,
    ):
        was_training = self.training
        self.eval()
        idx, prompt_lengths = self._prepare_generation_inputs(idx, tokenizer=tokenizer)
        batch_size, prompt_len = idx.size()

        max_new_tokens = int(max_new_tokens or getattr(self.config, "gen_len", 200))
        if prompt_len == 0:
            raise ValueError("Cannot generate from empty input")
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
        if max_new_tokens >= self.config.block_size:
            raise ValueError(
                f"max_new_tokens ({max_new_tokens}) must be smaller than block_size ({self.config.block_size})"
            )

        gamma = int(gamma or getattr(self.config, "speculative_gamma", 4))
        eos_id = getattr(self.config, "eos_id", None) if eos_id is None else eos_id
        use_speculative = (
            getattr(self.config, "enable_speculative", False)
            if use_speculative is None
            else bool(use_speculative)
        )
        use_speculative = use_speculative and draft_model is not None and batch_size == 1

        has_prompt_padding = any(length != prompt_len for length in prompt_lengths)
        if has_prompt_padding:
            if batch_size > 1:
                if caches is not None:
                    raise ValueError("Explicit caches are not supported for variable-length batched generation")
                sample_outputs = []
                for batch_idx, prompt_length in enumerate(prompt_lengths):
                    sample_outputs.append(
                        self.generate(
                            idx[batch_idx, :prompt_length].unsqueeze(0),
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            min_p=min_p,
                            repetition_penalty=repetition_penalty,
                            caches=None,
                            draft_model=draft_model,
                            gamma=gamma,
                            tokenizer=tokenizer,
                            return_dict=True,
                            include_prompt=include_prompt,
                            use_speculative=use_speculative,
                            eos_id=eos_id,
                        )
                    )
                return self._merge_generation_outputs(
                    sample_outputs,
                    tokenizer=tokenizer,
                    return_dict=return_dict,
                    include_prompt=include_prompt,
                )

            idx = idx[:, :prompt_lengths[0]]
            prompt_len = idx.size(1)
            prompt_lengths = [prompt_len]

        original_draft_training = None
        if draft_model is not None:
            original_draft_training = draft_model.training
            draft_model.eval()

        try:
            if use_speculative:
                return self._generate_speculative_naive(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    draft_model=draft_model,
                    gamma=gamma,
                    tokenizer=tokenizer,
                    return_dict=return_dict,
                    include_prompt=include_prompt,
                )

            n_layers = len(self.transformer.h)

            if prompt_len + max_new_tokens > self.config.block_size:
                idx = idx[:, -(self.config.block_size - max_new_tokens) :]
                prompt_lengths = [min(length, idx.size(1)) for length in prompt_lengths]
                prompt_len = idx.size(1)

            if caches is None:
                max_cache_len = min(
                    prompt_len + max_new_tokens,
                    getattr(self.config, "sliding_window", None) or self.config.block_size,
                )
                head_dim = self.config.n_embd // self.config.n_head
                caches = [
                    KVCache(
                        batch_size,
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
            freqs_cos = self.freqs_cos[:prompt_len]
            freqs_sin = self.freqs_sin[:prompt_len]
            for i, block in enumerate(self.transformer.h):
                x, caches[i] = block(x, freqs_cos, freqs_sin, kv_cache=caches[i])
            x = self.transformer.ln_f(x)
            logits = self._project_logits(x[:, -1, :])

            max_total_len = prompt_len + max_new_tokens
            all_so_far = torch.zeros((batch_size, max_total_len), dtype=torch.long, device=idx.device)
            all_so_far[:, :prompt_len] = idx
            current_len = prompt_len
            finished = torch.zeros(batch_size, dtype=torch.bool, device=idx.device)
            seen_mask = None
            if batch_size == 1:
                seen_mask = torch.zeros(self.config.vocab_size, dtype=torch.bool, device=idx.device)
                init_ids = idx[0]
                init_ids = init_ids[(init_ids >= 0) & (init_ids < self.config.vocab_size)]
                if init_ids.numel() > 0:
                    seen_mask[init_ids.unique()] = True

            while current_len < prompt_len + max_new_tokens:
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
                if eos_id is not None and finished.any():
                    idx_next = idx_next.clone()
                    idx_next[finished] = int(eos_id)
                all_so_far[:, current_len] = idx_next.squeeze(-1)
                if seen_mask is not None:
                    valid_token = idx_next.squeeze().clamp(0, self.config.vocab_size - 1)
                    seen_mask[valid_token] = True
                current_len += 1
                if eos_id is not None:
                    finished = finished | (idx_next.squeeze(-1) == int(eos_id))
                    if finished.all():
                        break
                x = self.transformer.wte(idx_next)
                pos = (current_len - 1) % self.freqs_cos.size(0)
                fc = self.freqs_cos[pos : pos + 1]
                fs = self.freqs_sin[pos : pos + 1]
                for i, block in enumerate(self.transformer.h):
                    x, caches[i] = block(x, fc, fs, kv_cache=caches[i])
                x = self.transformer.ln_f(x)
                logits = self._project_logits(x[:, -1, :])

            output = all_so_far[:, :current_len]
            return self._build_generation_output(
                output,
                prompt_lengths,
                tokenizer=tokenizer,
                return_dict=return_dict,
                include_prompt=include_prompt,
            )
        finally:
            self.train(was_training)
            if draft_model is not None and original_draft_training is not None:
                draft_model.train(original_draft_training)

    @torch.no_grad()
    def generate_stream(
        self,
        idx,
        max_new_tokens=None,
        temperature=1.0,
        top_k=0,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        tokenizer=None,
        return_dict=False,
        eos_id=None,
    ):
        was_training = self.training
        idx, prompt_lengths = self._prepare_generation_inputs(idx, tokenizer=tokenizer)
        max_new_tokens = int(max_new_tokens or getattr(self.config, "gen_len", 200))
        orig_prompt_len = idx.size(1)
        eos_id = getattr(self.config, "eos_id", None) if eos_id is None else eos_id
        all_so_far = idx

        try:
            self.eval()
            batch_size, prompt_len = idx.size()
            if prompt_len == 0:
                raise ValueError("Cannot generate from empty input")
            if max_new_tokens <= 0:
                raise ValueError(f"max_new_tokens must be > 0, got {max_new_tokens}")
            if max_new_tokens >= self.config.block_size:
                raise ValueError(
                    f"max_new_tokens ({max_new_tokens}) must be smaller than block_size ({self.config.block_size})"
                )

            if batch_size > 1:
                out = self.generate(
                    idx,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                    tokenizer=tokenizer,
                    return_dict=True,
                )
                generated_rows = out["generated_token_ids"]
                max_steps = max((len(row) for row in generated_rows), default=0)
                for step in range(max_steps):
                    token_ids = [row[step] if step < len(row) else None for row in generated_rows]
                    if return_dict:
                        texts = []
                        if tokenizer is not None:
                            texts = ["" if tok is None else tokenizer.decode([tok]) for tok in token_ids]
                        yield {
                            "step": step,
                            "token_ids": token_ids,
                            "texts": texts,
                            "finished": [tok is None or (eos_id is not None and tok == eos_id) for tok in token_ids],
                        }
                    else:
                        yield token_ids
                return

            n_layers = len(self.transformer.h)
            if prompt_len + max_new_tokens > self.config.block_size:
                idx = idx[:, -(self.config.block_size - max_new_tokens) :]
                prompt_len = idx.size(1)

            max_cache_len = min(
                prompt_len + max_new_tokens,
                getattr(self.config, "sliding_window", None) or self.config.block_size,
            )
            head_dim = self.config.n_embd // self.config.n_head
            caches = [
                KVCache(
                    batch_size,
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
            freqs_cos = self.freqs_cos[:prompt_len]
            freqs_sin = self.freqs_sin[:prompt_len]
            for i, block in enumerate(self.transformer.h):
                x, caches[i] = block(x, freqs_cos, freqs_sin, kv_cache=caches[i])
            x = self.transformer.ln_f(x)
            logits = self._project_logits(x[:, -1, :])

            all_so_far = idx.clone()
            seen_mask = torch.zeros(self.config.vocab_size, dtype=torch.bool, device=idx.device)
            init_ids = idx[0]
            init_ids = init_ids[(init_ids >= 0) & (init_ids < self.config.vocab_size)]
            if init_ids.numel() > 0:
                seen_mask[init_ids.unique()] = True

            for step in range(max_new_tokens):
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
                if 0 <= token_id < self.config.vocab_size:
                    seen_mask[token_id] = True
                all_so_far = torch.cat([all_so_far, idx_next], dim=1)
                if return_dict:
                    yield {
                        "step": step,
                        "token_id": token_id,
                        "text": "" if tokenizer is None else tokenizer.decode([token_id]),
                        "finished": eos_id is not None and token_id == int(eos_id),
                    }
                else:
                    yield token_id
                if eos_id is not None and token_id == int(eos_id):
                    break
                x = self.transformer.wte(idx_next)
                pos = (all_so_far.size(1) - 1) % self.freqs_cos.size(0)
                fc = self.freqs_cos[pos : pos + 1]
                fs = self.freqs_sin[pos : pos + 1]
                for i, block in enumerate(self.transformer.h):
                    x, caches[i] = block(x, fc, fs, kv_cache=caches[i])
                x = self.transformer.ln_f(x)
                logits = self._project_logits(x[:, -1, :])
        except Exception as exc:
            import logging
            generated_tokens = max(0, all_so_far.size(1) - orig_prompt_len)
            logging.error(f"Generation stream failed after {generated_tokens} tokens: {exc}")
            raise
        finally:
            self.train(was_training)
