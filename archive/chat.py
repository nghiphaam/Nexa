import os
import sys
import math
import json
import time
import argparse
import random
import multiprocessing as mp
import re
import copy
import uuid
import asyncio
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeout
from threading import Lock, RLock
from difflib import SequenceMatcher
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from lm import (
    NexaModel,
    Config,
    load_tokenizer,
    KVCache,
    make_amp_context,
    safe_load_model_state,
    EOS_TOKEN,
    PAD_TOKEN,
    SYS_TOKEN,
    USR_TOKEN,
    AST_TOKEN,
)

# ---------------------------------------------------------------------------
# NEXA 1.3 VERSION
# ---------------------------------------------------------------------------
NEXA_VERSION = "1.3"
SYSTEM_PROMPT = """You are Nexa, a concise and useful AI assistant.
- Answer clearly and directly.
- Be natural and conversational.
- Use concrete examples when possible.
- Avoid unnecessary repetition or vague answers.
- Use reasoning only when needed.
- If unsure, say you don't know instead of guessing.
"""
def _strip_reasoning(text):
    if "FINAL ANSWER:" in text:
        return text.split("FINAL ANSWER:", 1)[-1].strip()
    return text.strip()
SYSTEM_PREFIX = "System:\n"
USER_PREFIX = "User:\n"
ASSISTANT_PREFIX = "Assistant:\n"
MAX_TOOL_CHAIN = 3
DEFAULT_REASONING_BRANCHES = 1
DEFAULT_REASONING_STEPS = 0
MAX_REASONING_TOKENS = 0
MAX_REFLECTION_TOKENS = 0
FAIL_LOG_PATH = "data/fail_cases.jsonl"
SELF_IMPROVE_DATASET_PATH = "data/self_improve_dataset.jsonl"
NORMAL_QUERY_SEEDS = [
    "What is Python used for?",
    "Explain gravity in simple terms.",
    "How do I boil an egg?",
    "What is the capital of Japan?",
    "Give me three tips to study better.",
    "What does HTTP mean?",
]
def clone_critic_model(model, device):
    # FIX #6: Clone tensors to prevent shared reference side effects
    # Use state_dict to avoid zip order mismatch
    critic_model = copy.deepcopy(model)
    # Clone weight tensors (not just share) to prevent update side effects
    orig_state = model.state_dict()
    for name, param in critic_model.named_parameters():
        if name in orig_state:
            # FIX #6: Clone to avoid shared reference
            param.data = orig_state[name].detach().clone()
    critic_model = critic_model.to(device)
    critic_model.eval()
    for p in critic_model.parameters():
        p.requires_grad_(False)
    return critic_model
def _role_prefix(role):
    if role == "system":
        return SYSTEM_PREFIX
    if role == "user":
        return USER_PREFIX
    if role == "assistant":
        return ASSISTANT_PREFIX
    return ""
def _find_balanced_call_end(text, start_idx):
    depth = 1
    quote = None
    escaped = False
    for ii, ch in enumerate(text[start_idx:]):
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in {"'", '"'}:
            quote = ch
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return start_idx + ii
    return -1
def _safe_tool_eval(expr):
    import ast
    import math as _math
    allowed_names = {"abs", "min", "max", "sum", "len", "print", "math"}
    allowed_math_attrs = {
        "ceil",
        "fabs",
        "floor",
        "prod",
        "sqrt",
        "exp",
        "log",
        "log10",
        "sin",
        "cos",
        "tan",
    }
    allowed_nodes = (
        ast.Expression,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Dict,
        ast.UnaryOp,
        ast.UAdd,
        ast.USub,
        ast.Not,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.IfExp,
        ast.Subscript,
        ast.Slice,
        ast.Index,
        ast.Attribute,
    )
    try:
        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return "Error: unsafe expression or blocked tokens"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id == "range":
                        return "Error: range not allowed"
                    if node.func.id not in allowed_names:
                        return "Error: unsafe expression or blocked tokens"
                elif isinstance(node.func, ast.Attribute):
                    if (
                        not isinstance(node.func.value, ast.Name)
                        or node.func.value.id != "math"
                    ):
                        return "Error: unsafe expression or blocked tokens"
                    if node.func.attr not in allowed_math_attrs:
                        return "Error: unsafe expression or blocked tokens"
                else:
                    return "Error: unsafe expression or blocked tokens"
            if isinstance(node, ast.Name) and node.id not in allowed_names:
                return "Error: unsafe expression or blocked tokens"
            if isinstance(node, ast.Attribute):
                # Block access to dunder attributes to prevent object internals leak
                if node.attr.startswith("__"):
                    return "Error: unsafe expression or blocked tokens"
                if not isinstance(node.value, ast.Name) or node.value.id != "math":
                    return "Error: unsafe expression or blocked tokens"
                if node.attr not in allowed_math_attrs:
                    return "Error: unsafe expression or blocked tokens"
        captured = []
        def _capturing_print(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            end = kwargs.get("end", "\n")
            captured.append(sep.join(str(a) for a in args) + end)
        env = {
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "print": _capturing_print,
            "math": _math,
        }
        try:
            result = eval(compile(tree, "<tool>", "eval"), {"__builtins__": {}}, env)
            output = "".join(captured).strip()
            if output:
                return output
            return str(result)
        except Exception as e:
            return f"Error: {e}"
    except Exception:
        return "Error: unsafe expression or blocked tokens"
def _split_reasoning_answer(content):
    if "FINAL ANSWER:" not in content:
        return None, None
    thought_text, answer_text = content.split("FINAL ANSWER:", 1)
    return thought_text.strip(), answer_text.strip()
def _build_reasoning_scaffold(
    thought_text, answer_text, score_text="1.0", refine_text=None
):
    thought_text = (thought_text or answer_text or "").strip()
    answer_text = (answer_text or "").strip()
    refine_text = (refine_text if refine_text is not None else answer_text).strip()
    return (
        f"Thought:\n{thought_text}\n\n"
        f"Score:\n{score_text}\n\n"
        f"Refine:\n{refine_text}\n\n"
        f"FINAL ANSWER:{answer_text}"
    )
def _encode_assistant_content(tokenizer, content):
    if all(
        marker in content
        for marker in ("Thought:\n", "\nScore:\n", "\nRefine:\n", "FINAL ANSWER:")
    ):
        scaffold_prefix, answer_text = content.split("FINAL ANSWER:", 1)
        prefix_ids = tokenizer.encode(scaffold_prefix).ids
        answer_prefix_ids = tokenizer.encode("FINAL ANSWER:").ids
        answer_ids = tokenizer.encode(answer_text).ids
        ids = prefix_ids + answer_prefix_ids + answer_ids
        reasoning_mask = [True] * len(prefix_ids) + [False] * (
            len(answer_prefix_ids) + len(answer_ids)
        )
        return ids, reasoning_mask
    thought_text, answer_text = _split_reasoning_answer(content)
    if answer_text is None:
        ids = tokenizer.encode(content).ids
        return ids, [False] * len(ids)
    thought_ids = tokenizer.encode(f"Thought:\n{thought_text}\n\n").ids
    score_ids = tokenizer.encode("Score:\n1.0\n\n").ids
    refine_ids = tokenizer.encode(f"Refine:\n{answer_text}\n\n").ids
    answer_prefix_ids = tokenizer.encode("FINAL ANSWER:").ids
    answer_ids = tokenizer.encode(answer_text).ids
    ids = thought_ids + score_ids + refine_ids + answer_prefix_ids + answer_ids
    reasoning_mask = [True] * (len(thought_ids) + len(score_ids) + len(refine_ids))
    reasoning_mask.extend([False] * (len(answer_prefix_ids) + len(answer_ids)))
    return ids, reasoning_mask
def format_chat(
    messages, tokenizer, add_generation_prompt=True, return_reasoning_mask=False
):
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    use_special_roles = getattr(tokenizer, "supports_special_role_tokens", True)
    # FIX #5: Validate messages and warn about drops (with stacklevel)
    if not messages:
        if return_reasoning_mask:
            return [], [], []
        return [], []
    original_len = len(messages)
    messages = [m for m in messages if m and isinstance(m, dict) and "role" in m and "content" in m]
    if len(messages) < original_len:
        import warnings
        warnings.warn(
            f"format_chat: dropped {original_len - len(messages)} malformed messages",
            stacklevel=2
        )
    if use_special_roles:
        sys_id = tokenizer.token_to_id(SYS_TOKEN)
        usr_id = tokenizer.token_to_id(USR_TOKEN)
        ast_id = tokenizer.token_to_id(AST_TOKEN)
    all_ids = []
    assistant_mask = []
    reasoning_mask = []
    for msg in messages:
        role = msg["role"]
        if use_special_roles:
            if role == "assistant" and return_reasoning_mask:
                content_ids, content_reasoning = _encode_assistant_content(
                    tokenizer, msg["content"]
                )
            else:
                content_ids = tokenizer.encode(msg["content"]).ids
                content_reasoning = [False] * len(content_ids)
            if role == "system":
                header = [sys_id]
            elif role == "user":
                header = [usr_id]
            elif role == "assistant":
                header = [ast_id]
                all_ids.extend(header + content_ids + [eos_id])
                assistant_mask.extend([False] * len(header))
                assistant_mask.extend([True] * len(content_ids))
                assistant_mask.append(True)
                reasoning_mask.extend([False] * len(header))
                reasoning_mask.extend(content_reasoning)
                reasoning_mask.append(False)
                continue
            else:
                header = []
            all_ids.extend(header + content_ids)
            assistant_mask.extend([False] * (len(header) + len(content_ids)))
            reasoning_mask.extend([False] * (len(header) + len(content_ids)))
        else:
            header_ids = tokenizer.encode(_role_prefix(role)).ids
            if role == "assistant" and return_reasoning_mask:
                content_ids, content_reasoning = _encode_assistant_content(
                    tokenizer, msg["content"]
                )
            else:
                content_ids = tokenizer.encode(msg["content"]).ids
                content_reasoning = [False] * len(content_ids)
            all_ids.extend(header_ids + content_ids + [eos_id])
            assistant_mask.extend([False] * len(header_ids))
            assistant_mask.extend([role == "assistant"] * len(content_ids))
            assistant_mask.append(role == "assistant")
            reasoning_mask.extend([False] * len(header_ids))
            reasoning_mask.extend(
                content_reasoning if role == "assistant" else [False] * len(content_ids)
            )
            reasoning_mask.append(False)
    if add_generation_prompt:
        if use_special_roles:
            all_ids.append(ast_id)
            assistant_mask.append(False)
            reasoning_mask.append(False)
            thought_ids = tokenizer.encode("Thought:\n").ids
        else:
            thought_ids = tokenizer.encode("Assistant:\nThought:\n").ids
        all_ids.extend(thought_ids)
        assistant_mask.extend([False] * len(thought_ids))
        reasoning_mask.extend([False] * len(thought_ids))
    if return_reasoning_mask:
        return all_ids, assistant_mask, reasoning_mask
    return all_ids, assistant_mask
def build_chat_prompt(history, user_msg, system_prompt=SYSTEM_PROMPT):
    messages = [{"role": "system", "content": system_prompt}]
    for u, b in history:
        messages.append({"role": "user", "content": u})
        if b:
            messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": user_msg})
    return messages
def _build_training_trace_messages(messages):
    traced = []
    for msg in messages:
        traced.append(dict(msg))
        if msg.get("role") != "assistant":
            continue
        thought_text, answer_text = _split_reasoning_answer(msg.get("content", ""))
        if not answer_text:
            continue
        traced.append(
            {
                "role": "assistant",
                "content": (
                    f"Critic:\nChecked reasoning for consistency.\n"
                    f"Score:\n0.8\n\n"
                    f"Refine:\n{answer_text}\n\n"
                    f"FINAL ANSWER:{answer_text}"
                ),
            }
        )
        traced.append(
            {
                "role": "assistant",
                "content": (
                    f"Tool result: verification unavailable\nFINAL ANSWER:{answer_text}"
                ),
            }
        )
        break
    return traced
class LoRALinear(nn.Module):
    def __init__(self, original, rank=8, alpha=16.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(original.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, original.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.original.parameters():
            p.requires_grad = False
    def forward(self, x):
        return self.original(x) + (x @ self.lora_A @ self.lora_B) * self.scaling
def apply_lora(model, rank=8, alpha=16.0, target_modules=None):
    if target_modules is None:
        target_modules = ["wq", "wk", "wv", "c_proj", "w1", "w2", "w3"]
    for p in model.parameters():
        p.requires_grad = False
    lora_count = 0
    for block in model.transformer.h:
        for name in ["wq", "wk", "wv", "c_proj"]:
            if name in target_modules and hasattr(block.attn, name):
                orig = getattr(block.attn, name)
                if isinstance(orig, nn.Linear):
                    setattr(block.attn, name, LoRALinear(orig, rank, alpha))
                    lora_count += 1
        for name in ["w1", "w2", "w3"]:
            if name in target_modules and hasattr(block.mlp, name):
                orig = getattr(block.mlp, name)
                if isinstance(orig, nn.Linear):
                    setattr(block.mlp, name, LoRALinear(orig, rank, alpha))
                    lora_count += 1
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {lora_count} layers, rank={rank}, alpha={alpha}")
    print(f"  Trainable: {n_train:,} / {n_total:,} ({100 * n_train / n_total:.2f}%)")
    return model
def save_lora(model, path):
    state = {
        n: p.data
        for n, p in model.named_parameters()
        if p.requires_grad and ("lora_A" in n or "lora_B" in n)
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    print(f"Saved LoRA: {path} ({len(state)} tensors)")
def load_lora(model, path):
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=False
    )
    print(f"Loaded LoRA: {path}")
    return model
def merge_lora(model):
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # FIX #6: Prevent double merge
            if getattr(module, "_merged", False):
                continue
            delta = (module.lora_A @ module.lora_B) * module.scaling
            # FIX #7: Use float precision to prevent accumulation error
            orig_weight = module.original.weight.data
            module.original.weight.data = (
                orig_weight.float() + delta.T.float()
            ).to(orig_weight.dtype)
            module.lora_A.data.zero_()
            module.lora_B.data.zero_()
            module._merged = True
            merged += 1
    print(f"Merged {merged} LoRA layers into base weights")
    return model
class VectorMemory:
    MAX_DB = 500
    def __init__(self, wte_module, emb_dim=None):
        self.wte = wte_module
        self.lock = Lock()
        self.db = []
        raw_dim = wte_module.weight.shape[1]
        project_dim = emb_dim or max(64, raw_dim // 8)
        self.head = nn.Linear(raw_dim, project_dim, bias=False)
        nn.init.orthogonal_(self.head.weight)
        # NEXA 1.3: Embedding cache for faster retrieval
        self._emb_cache_tensor = None  # [N, project_dim] stacked embeddings
        self._emb_cache_dirty = True
    def _sync_head_device(self, device):
        if self.head.weight.device != device:
            self.head = self.head.to(device)
    def project(self, x):
        with torch.no_grad():
            raw = x.detach()
            self._sync_head_device(raw.device)
            proj = self.head(raw)
            return F.normalize(proj, p=2, dim=-1).detach()
    @torch.no_grad()
    def _pool_raw(self, ids):
        if not ids:
            return None
        device = self.wte.weight.device
        t = torch.tensor([ids], dtype=torch.long, device=device)
        tok = self.wte(t).detach()  
        T = tok.size(1)
        recency = torch.linspace(0.8, 1.2, T, device=device, dtype=tok.dtype).view(1, T)
        salience = tok.norm(dim=-1).clamp_min(1e-6)
        weights = salience * recency
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (tok * weights.unsqueeze(-1)).sum(dim=1).detach()
    @torch.no_grad()
    def embed(self, ids):
        raw = self._pool_raw(ids)
        if raw is None:
            return None
        # FIX #3: Keep on GPU to avoid CPU/GPU transfer overhead
        return self.project(raw)
    def add(self, text, ids, timestamp=None, memory_type="conversation"):
        import time as _time
        emb = self.embed(ids)
        raw_state = self._pool_raw(ids)
        if emb is not None and raw_state is not None:
            with self.lock:
                # FIX #6: Dedup - check last few entries (not just last one)
                for i in range(min(3, len(self.db))):
                    if self.db[-(i+1)][2] == text:
                        return
                if len(self.db) >= self.MAX_DB:
                    self.db.pop(0)
                ts = timestamp if timestamp is not None else _time.monotonic()
                # FIX #3: Store on CPU to save GPU memory
                self.db.append((emb.cpu(), raw_state.cpu(), text, ts, memory_type))
                # NEXA 1.3: Invalidate stacked cache
                self._emb_cache_dirty = True

    def _rebuild_emb_cache(self, device):
        """NEXA 1.3: Rebuild stacked embedding tensor for batch cosine similarity."""
        if not self.db:
            self._emb_cache_tensor = None
            self._emb_cache_dirty = False
            return
        embs = [entry[0].to(device) for entry in self.db]  # list of [1, D]
        self._emb_cache_tensor = torch.cat(embs, dim=0)  # [N, D]
        self._emb_cache_dirty = False

    def retrieve_kv(self, query_ids, decay=0.05, top_k=None, min_score=0.2):
        import time as _time
        with self.lock:
            if not self.db:
                return []
            # NEXA 1.2: Clone tensors to prevent CUDA/CPU race condition
            db_snapshot = [
                (emb.clone(), raw_state.clone(), text, ts, memory_type)
                for emb, raw_state, text, ts, memory_type in self.db
            ]
            now = _time.monotonic()
            # NEXA 1.3: Check if cache needs rebuild
            cache_dirty = self._emb_cache_dirty
        top_k = top_k or min(4, max(1, len(db_snapshot) // 5))
        q_emb = self.embed(query_ids)
        if q_emb is None:
            return []

        # NEXA 1.3: Batch similarity with stacked tensor for O(1) vs O(N) individual calls
        device = q_emb.device
        if cache_dirty:
            with self.lock:
                self._rebuild_emb_cache(device)
        cache = self._emb_cache_tensor
        if cache is not None and cache.device != device:
            cache = cache.to(device)

        type_bonus = {"fact": 1.08, "tool_result": 1.04, "conversation": 1.0}
        hits = []
        if cache is not None and cache.shape[0] == len(db_snapshot):
            # Fast path: batch cosine similarity
            sims = F.cosine_similarity(q_emb.unsqueeze(0), cache, dim=-1)  # [N]
            sims = sims.clamp(-1.0, 1.0).cpu().tolist()
            for idx, (sim, (emb, _raw_state, text, ts, memory_type)) in enumerate(
                zip(sims, db_snapshot)
            ):
                age = now - ts
                score = sim * math.exp(-decay * age) * type_bonus.get(memory_type, 1.0)
                if score >= min_score:
                    hits.append((score, memory_type, text))
        else:
            # Fallback: individual similarity
            for emb, _raw_state, text, ts, memory_type in db_snapshot:
                sim = F.cosine_similarity(q_emb, emb.to(device), dim=-1).item()
                sim = max(-1.0, min(1.0, sim))
                age = now - ts
                score = sim * math.exp(-decay * age) * type_bonus.get(memory_type, 1.0)
                if score >= min_score:
                    hits.append((score, memory_type, text))

        hits.sort(key=lambda x: x[0], reverse=True)
        if not hits and db_snapshot:
            _emb, _raw_state, text, _ts, memory_type = db_snapshot[-1]
            return [(0.0, memory_type, text)]
        return hits[:top_k]
    def retrieve(self, query_ids, decay=0.05):
        hits = self.retrieve_kv(query_ids, decay=decay)
        if not hits:
            return ""
        memory_text = "\n".join(
            [
                f"[Memory:{memory_type}|score={score:.2f}] {text}"
                for score, memory_type, text in hits
            ]
        )
        return memory_text[:1000]
    def retrieve_state(self, query_ids, decay=0.05, device=None):
        import time as _time
        with self.lock:
            if not self.db:
                return None
            # NEXA 1.2: Clone tensors to prevent CUDA/CPU race condition
            db_snapshot = [
                (emb.clone(), raw_state.clone(), text, ts, memory_type)
                for emb, raw_state, text, ts, memory_type in self.db
            ]
            now = _time.monotonic()
        q_emb = self.embed(query_ids)
        if q_emb is None:
            return None
        type_bonus = {"fact": 1.08, "tool_result": 1.04, "conversation": 1.0}
        scored = []
        for emb, raw_state, _text, ts, memory_type in db_snapshot:
            # NEXA 1.2: Use F.cosine_similarity for guaranteed correctness
            sim = F.cosine_similarity(q_emb, emb, dim=-1).item()
            sim = max(-1.0, min(1.0, sim))
            age = now - ts
            score = sim * math.exp(-decay * age) * type_bonus.get(memory_type, 1.0)
            if score < 0.15:
                continue
            scored.append((score, raw_state))
        if not scored:
            return None
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: min(4, len(scored))]
        # FIX #2: Match dtype with model weights
        target_device = device or self.wte.weight.device
        target_dtype = self.wte.weight.dtype
        weights = torch.tensor([max(s, 1e-4) for s, _ in top], dtype=target_dtype, device=target_device)
        weights = weights / weights.sum().clamp_min(1e-6)
        state = sum(w * raw.squeeze(0).to(dtype=target_dtype, device=target_device) for w, (_, raw) in zip(weights, top))

        # NEXA 1.2: Check for zero vector before normalize to prevent NaN
        state_norm = torch.norm(state)
        if state_norm < 1e-6:
            return None

        # FIX #3: Add eps to prevent NaN from zero vectors
        state = F.normalize(state, dim=-1, eps=1e-6)
        return state
    def query_state(self, query_ids, device=None):
        raw = self._pool_raw(query_ids)
        if raw is None:
            return None
        return F.normalize(raw.squeeze(0).float(), dim=-1).to(
            device or self.wte.weight.device
        )
    def to(self, device):
        self.head = self.head.to(device)
        # NEXA 1.3: Invalidate stacked embedding cache on device change
        self._emb_cache_dirty = True
        self._emb_cache_tensor = None
        return self
# ---------------------------------------------------------------------------
# NEXA 1.3: DYNAMIC BATCHING SCHEDULER
# ---------------------------------------------------------------------------

@dataclass
class BatchRequest:
    """A single inference request in the batch queue."""
    request_id: str
    input_ids: List[int]
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float
    min_p: float
    repetition_penalty: float
    result_queue: queue.Queue = field(default_factory=queue.Queue)
    session_id: str = ""
    created_at: float = field(default_factory=time.time)


class DynamicBatchScheduler:
    """
    Collects incoming requests and merges them into batched forward passes.

    Workflow:
    1. Requests arrive via submit() → put into pending queue
    2. Background thread polls queue with timeout
    3. When batch_size reached OR timeout expires → form batch
    4. Pad sequences to max_len, build attention mask
    5. Run single forward pass for the whole batch
    6. Scatter results back to individual request queues

    Performance: throughput x2-x5 vs sequential, GPU utilization ↑↑
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_batch_size=8,
        batch_timeout_ms=50,
        max_seq_len=2048,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.max_seq_len = max_seq_len
        self._pending = queue.Queue()
        self._running = False
        self._thread = None
        self._lock = Lock()
        self._stats = {"batches": 0, "total_requests": 0, "avg_batch_size": 0.0}

    def start(self):
        """Start the background batching thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._batch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background batching thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def submit(self, request: BatchRequest):
        """Submit a request for batched inference."""
        self._pending.put(request)

    def _batch_loop(self):
        """Main loop: collect requests, form batches, run inference."""
        while self._running:
            batch = []
            # Wait for first request
            try:
                first = self._pending.get(timeout=0.1)
                batch.append(first)
            except queue.Empty:
                continue

            # Collect more requests within timeout
            deadline = time.time() + self.batch_timeout
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    remaining = max(0.001, deadline - time.time())
                    req = self._pending.get(timeout=remaining)
                    batch.append(req)
                except queue.Empty:
                    break

            if batch:
                try:
                    self._process_batch(batch)
                except Exception as e:
                    # On error, send error to all waiting requests
                    for req in batch:
                        req.result_queue.put({"error": str(e), "tokens": []})

    @torch.no_grad()
    def _process_batch(self, batch: List[BatchRequest]):
        """Process a batch of requests with padding and single forward pass."""
        batch_size = len(batch)

        # Update stats
        with self._lock:
            self._stats["batches"] += 1
            self._stats["total_requests"] += batch_size
            self._stats["avg_batch_size"] = (
                self._stats["total_requests"] / self._stats["batches"]
            )

        # Find max length for padding
        all_input_ids = [req.input_ids[-self.max_seq_len:] for req in batch]
        max_len = max(len(ids) for ids in all_input_ids)

        # Pad sequences and create attention mask
        pad_id = self.tokenizer.token_to_id(PAD_TOKEN)
        padded_ids = []
        attention_masks = []

        for ids in all_input_ids:
            pad_len = max_len - len(ids)
            # Left-pad for causal LM
            padded = [pad_id] * pad_len + ids
            mask = [0] * pad_len + [1] * len(ids)
            padded_ids.append(padded)
            attention_masks.append(mask)

        # Create tensors
        input_tensor = torch.tensor(padded_ids, dtype=torch.long, device=self.device)
        attn_mask = torch.tensor(attention_masks, dtype=torch.bool, device=self.device)

        # Forward pass (prefill)
        amp_dtype = (
            next(self.model.parameters()).dtype
            if self.device == "cuda"
            else torch.float32
        )
        amp_ctx = make_amp_context(self.device, amp_dtype)

        with amp_ctx:
            x = self.model.transformer.wte(input_tensor)
            T = input_tensor.size(1)
            freqs_cos = self.model.freqs_cos[:T]
            freqs_sin = self.model.freqs_sin[:T]

            for block in self.model.transformer.h:
                x, _ = block(x, freqs_cos, freqs_sin, None)

            x = self.model.transformer.ln_f(x)
            # Get logits for last real token of each sequence
            logits = self.model.lm_head(x)

        # For each request, get the logits at their actual last position
        for i, req in enumerate(batch):
            actual_len = len(all_input_ids[i])
            last_pos = max_len - 1  # last position in padded sequence
            req_logits = logits[i, last_pos, :].unsqueeze(0)

            # Sample first token
            token_id = self.model._sample_token(
                req_logits,
                input_tensor[i:i+1],
                req.temperature,
                req.top_k,
                req.top_p,
                req.min_p,
                req.repetition_penalty,
                track_entropy=False,
            ).item()

            req.result_queue.put({
                "token_id": token_id,
                "logits_shape": list(req_logits.shape),
            })

    @property
    def stats(self):
        with self._lock:
            return dict(self._stats)


# ---------------------------------------------------------------------------
# NEXA 1.3: MULTI-SESSION MANAGER
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages multiple ChatSession instances, one per user/session_id.

    Features:
    - Auto-create on first access
    - TTL-based expiry (default 30 minutes)
    - Thread-safe access
    - Max sessions limit to prevent memory leak
    - LRU eviction when limit reached
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        critic_model=None,
        max_sessions=50,
        session_ttl=1800,  # 30 minutes
        **default_kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.critic_model = critic_model
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl
        self.default_kwargs = default_kwargs
        self._sessions: OrderedDict[str, dict] = OrderedDict()
        self._lock = RLock()

    def get_or_create(self, session_id: str) -> "ChatSession":
        """Get existing session or create new one. Thread-safe."""
        with self._lock:
            self._cleanup_expired()

            if session_id in self._sessions:
                entry = self._sessions[session_id]
                entry["last_access"] = time.time()
                # Move to end (most recently used)
                self._sessions.move_to_end(session_id)
                return entry["session"]

            # Evict oldest if at capacity
            while len(self._sessions) >= self.max_sessions:
                oldest_key, oldest_entry = next(iter(self._sessions.items()))
                oldest_entry["session"].close()
                del self._sessions[oldest_key]

            # Create new session
            session = ChatSession(
                self.model,
                self.tokenizer,
                self.device,
                critic_model=self.critic_model,
                **self.default_kwargs,
            )

            self._sessions[session_id] = {
                "session": session,
                "created_at": time.time(),
                "last_access": time.time(),
            }
            return session

    def get(self, session_id: str) -> Optional["ChatSession"]:
        """Get session without creating. Returns None if not found."""
        with self._lock:
            if session_id in self._sessions:
                entry = self._sessions[session_id]
                entry["last_access"] = time.time()
                self._sessions.move_to_end(session_id)
                return entry["session"]
            return None

    def remove(self, session_id: str):
        """Remove and cleanup a specific session."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["session"].close()
                del self._sessions[session_id]

    def _cleanup_expired(self):
        """Remove sessions that have exceeded TTL."""
        now = time.time()
        expired = [
            sid for sid, entry in self._sessions.items()
            if now - entry["last_access"] > self.session_ttl
        ]
        for sid in expired:
            self._sessions[sid]["session"].close()
            del self._sessions[sid]

    def list_sessions(self) -> List[dict]:
        """List all active sessions with metadata."""
        with self._lock:
            result = []
            for sid, entry in self._sessions.items():
                result.append({
                    "session_id": sid,
                    "created_at": entry["created_at"],
                    "last_access": entry["last_access"],
                    "turn_count": entry["session"].turn_count,
                    "age_s": time.time() - entry["created_at"],
                })
            return result

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def close_all(self):
        """Shutdown all sessions."""
        with self._lock:
            for entry in self._sessions.values():
                try:
                    entry["session"].close()
                except Exception:
                    pass
            self._sessions.clear()


class ChatSession:
    def __init__(
        self,
        model,
        tokenizer,
        device,
        temperature=0.6,
        top_k=40,
        top_p=0.9,
        min_p=0.05,
        repetition_penalty=1.1,
        max_tokens=200,
        fast_mode=True,
        critic_model=None,
    ):
        self.model = model
        self.critic_model = (
            critic_model
            if critic_model is not None
            else clone_critic_model(model, device)
        )
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.max_tokens = max_tokens
        self.fast_mode = fast_mode
        self.history = []
        self.all_token_ids = []
        self.global_pos = 0  
        self.caches = None
        self.memory = VectorMemory(model.transformer.wte).to(device)
        self.memory.head.requires_grad_(False)  
        self.memory.head.eval()
        self.reflection_enabled = False
        self.reasoning_enabled = False
        self.reasoning_branches = DEFAULT_REASONING_BRANCHES
        self.reasoning_steps = DEFAULT_REASONING_STEPS
        self.max_tool_chain = MAX_TOOL_CHAIN
        self._has_reflected = False  
        self._tool_calls = 0  
        self.turn_count = 0
        self.last_telemetry = {}
        self._auto_fast_turns = 0
        self._force_reasoning_turns = 0
        self._good_runtime_streak = 0
        self._tool_executor = None
        self._ensure_executor()
        self._reset_cache()
    def set_mode(self, fast_mode):
        self.fast_mode = bool(fast_mode)
        if self.fast_mode:
            self.reasoning_enabled = False
            self.reflection_enabled = False
        else:
            self.reasoning_enabled = True
            self.reflection_enabled = True
    def _ensure_executor(self):
        executor = getattr(self, "_tool_executor", None)
        # FIX #5: Check if executor is actually alive, not just _shutdown flag
        if executor is None or getattr(executor, "_shutdown", False):
            # Clean up old executor first
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            self._tool_executor = ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn")
            )
        return self._tool_executor

    def _cleanup_executor(self):
        # FIX #6: Ensure executor is properly cleaned up
        executor = getattr(self, "_tool_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._tool_executor = None
    def _reset_cache(self):
        n_layers = len(self.model.transformer.h)
        sw = (
            getattr(self.model.config, "sliding_window", None)
            or self.model.config.block_size
        )
        self.max_cache_len = min(self.model.config.block_size, sw)
        head_dim = self.model.config.n_embd // self.model.config.n_head
        dtype = next(self.model.parameters()).dtype
        self.caches = [
            KVCache(
                1,
                self.max_cache_len,
                self.model.config.n_kv_head,
                head_dim,
                self.device,
                dtype,
                n_global_tokens=getattr(self.model.config, "n_global_tokens", 0),
            )
            for _ in range(n_layers)
        ]
        self.all_token_ids = []
        self.all_so_far = torch.zeros(
            (1, self.max_cache_len), dtype=torch.long, device=self.device
        )
        self.current_len = 0
        self.freqs_cos = self.model.freqs_cos.to(self.device)
        self.freqs_sin = self.model.freqs_sin.to(self.device)
        self.model._current_entropy = None
        self.model._current_entropy_norm = None
        self.model._entropy_ema = None
        self.model._entropy_var_ema = None
        self.model._reflect_cooldown = 0
        self._has_reflected = False
        if (
            not hasattr(self, "seen_mask")
            or self.seen_mask.numel() != self.model.config.vocab_size
        ):
            self.seen_mask = torch.zeros(
                self.model.config.vocab_size, dtype=torch.bool, device=self.device
            )
        else:
            self.seen_mask.zero_()
    def _mark_seen(self, ids):
        if not ids:
            return
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        # FIX #5: Clamp to valid vocab range and warn if truncated
        valid_mask = (ids_t >= 0) & (ids_t < self.seen_mask.numel())
        if not valid_mask.all():
            import warnings
            # NEXA 1.2: Fix missing .item() in warning format
            warnings.warn(f"_mark_seen: {(~valid_mask).sum().item()} tokens out of vocab range")
        ids_t = ids_t[valid_mask]
        # NEXA 1.2: Return early if all ids invalid to prevent silent skip
        if ids_t.numel() == 0:
            return
        self.seen_mask[ids_t.unique()] = True
    def _rebuild_seen_mask(self):
        self.seen_mask.zero_()
        self._mark_seen(self.all_token_ids)
    def _rope_slice(self, start, length, device=None):
        max_len = self.freqs_cos.size(0)
        pos_range = (
            torch.arange(start, start + length, device=device or self.device) % max_len
        )
        return self.freqs_cos[pos_range], self.freqs_sin[pos_range]
    def _cap_hidden_suffix(self, prefix_ids, suffix_ids, total_budget, hard_cap):
        if not suffix_ids:
            return []
        suffix_budget = max(0, min(hard_cap, total_budget - len(prefix_ids)))
        if suffix_budget <= 0:
            return []
        return suffix_ids[-suffix_budget:]
    def _cap_runtime_injection(self, token_ids, hard_cap, reserve_tokens=1):
        if not token_ids:
            return []
        available = min(
            hard_cap,
            max(0, self.max_cache_len - self.current_len - reserve_tokens),
            max(0, self.model.config.block_size - self.current_len - reserve_tokens),
        )
        if available <= 0:
            return []
        return token_ids[-available:]
    def _inject_memory_state(self, x, memory_state, memory_query_state=None):
        return self.model._apply_memory_state(x, memory_state, memory_query_state)
    def _assert_state_sync(self):
        assert self.current_len == len(self.all_token_ids), (
            f"state desync: current_len={self.current_len} all_token_ids={len(self.all_token_ids)}"
        )
    def _log_fail_case(self, user_msg, response, reason, extra=None):
        extra = extra or {}
        tags = []
        lowered_response = (response or "").lower()
        if reason == "tool_failure":
            tags.append("tool_misuse")
        if any(
            tok in lowered_response
            for tok in ("therefore", "because", "step", "reason")
        ) and reason in {"low_critic_score", "tool_failure"}:
            tags.append("reasoning_error")
        if any(ch.isdigit() for ch in response or "") and any(
            sym in user_msg for sym in ("+", "-", "*", "/", "=", "%")
        ):
            tags.append("math_error")
        if (
            reason == "low_critic_score"
            and "tool_misuse" not in tags
            and "math_error" not in tags
        ):
            tags.append("hallucination")
        os.makedirs(os.path.dirname(FAIL_LOG_PATH) or ".", exist_ok=True)
        row = {
            "ts": time.time(),
            "reason": reason,
            "tags": tags,
            "user": user_msg,
            "response": response,
            "extra": extra,
        }
        try:
            with open(FAIL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass
    def _build_answer_prompt(self, user_msg, memory_context=""):
        if memory_context:
            user_msg = f"[Memory Context]\n{memory_context}\n\nUser: {user_msg}"
        prompt = f"User request:\n{user_msg}\n\n"
        prompt += "Answer clearly and directly.\nFINAL ANSWER:\n"
        return prompt
    def _generate_text(
        self,
        prompt_text,
        max_new_tokens=96,
        temperature=None,
        top_k=None,
        top_p=None,
        min_p=None,
        repetition_penalty=None,
        head="main",
        model_override=None,
    ):
        prompt_ids = self.tokenizer.encode(prompt_text).ids
        active_model = model_override or self.model
        max_prompt_len = max(1, active_model.config.block_size - max_new_tokens)
        prompt_ids = prompt_ids[-max_prompt_len:]
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        saved_state = (
            getattr(active_model, "_current_entropy", None),
            getattr(active_model, "_current_entropy_norm", None),
            getattr(active_model, "_entropy_ema", None),
            getattr(active_model, "_entropy_var_ema", None),
            getattr(active_model, "_reflect_cooldown", 0),
        )
        streamed = []
        for tok in active_model.generate_stream(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_k=self.top_k if top_k is None else top_k,
            top_p=self.top_p if top_p is None else top_p,
            min_p=self.min_p if min_p is None else min_p,
            repetition_penalty=self.repetition_penalty
            if repetition_penalty is None
            else repetition_penalty,
            head=head,
        ):
            streamed.append(tok)
        (
            active_model._current_entropy,
            active_model._current_entropy_norm,
            active_model._entropy_ema,
            active_model._entropy_var_ema,
            active_model._reflect_cooldown,
        ) = saved_state
        return self.tokenizer.decode(streamed, skip_special_tokens=False)
    def _truncate_generated(self, text, stop_markers):
        cut = len(text)
        for marker in stop_markers:
            idx = text.find(marker)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut].strip()
    def _visible_response(self, text):
        if self.fast_mode:
            return _strip_reasoning(text)
        if "FINAL ANSWER:" in text:
            text = text.split("FINAL ANSWER:", 1)[1]
        leak_markers = ["Plan:\n", "Best Thought:\n", "Critic:\n", "Critic Score:"]
        for marker in leak_markers:
            idx = text.find(marker)
            if 0 <= idx < 120:
                text = text[:idx]
        return text.lstrip()
    def _extract_score(self, critique):
        labelled = re.search(r"Score:\s*([01](?:\.\d+)?)", critique)
        if labelled:
            return max(0.0, min(1.0, float(labelled.group(1))))
        floats = re.findall(r"(?<![\d.])([01](?:\.\d+)?)(?![\d.])", critique)
        if floats:
            return max(0.0, min(1.0, float(floats[0])))
        percents = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", critique)
        if percents:
            return max(0.0, min(1.0, float(percents[0]) / 100.0))
        return 0.3
    def _extract_final_answer(self, thought):
        if "FINAL ANSWER:" in thought:
            return thought.split("FINAL ANSWER:", 1)[1].strip()
        return thought.strip().splitlines()[-1].strip() if thought.strip() else ""
    def _reasoning_budget(self, user_msg):
        complexity = len(user_msg)
        lowered = user_msg.lower()
        hard_markers = (
            "why",
            "prove",
            "analyze",
            "compare",
            "debug",
            "design",
            "tradeoff",
            "derive",
            "plan",
        )
        if complexity < 140 or not any(marker in lowered for marker in hard_markers):
            return 1, 1, 1
        if complexity < 260:
            return min(2, self.reasoning_branches), 1, 1
        return (
            min(2, self.reasoning_branches),
            min(2, self.reasoning_steps),
            min(2, self.reasoning_branches),
        )
    def _generate_critic(self, prompt):
        return self._generate_text(
            prompt,
            max_new_tokens=80,
            temperature=0.2,
            top_k=30,
            top_p=0.9,
            min_p=0.0,
            repetition_penalty=1.0,
            head="critic",
            model_override=self.critic_model,
        )
    def _critic_score(self, user_msg, plan, thought):
        critic_prompt = (
            "You are a skeptical reasoning verifier.\n"
            "Assume the answer is WRONG.\n"
            "Find the strongest flaw first.\n"
            "Only give high score if no critical issue exists.\n"
            "Prefer factual correctness over style.\n"
            f"User problem:\n{user_msg}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Candidate reasoning:\n{thought}\n\n"
            "Find the strongest flaw first, then estimate confidence.\n"
            "Return exactly:\n"
            "Score: <0.0-1.0>\n"
            "Critique: <one short paragraph>\n"
        )
        critique = self._generate_critic(critic_prompt)
        critique = self._truncate_generated(
            critique, ["\n\nUser", "\nUser:", "\nAssistant:"]
        )
        parsed_score = self._extract_score(critique)
        critic_ids = self.tokenizer.encode(critic_prompt).ids[
            -self.critic_model.config.block_size :
        ]
        critic_x = torch.tensor([critic_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            _logits, _loss, aux = self.critic_model(critic_x, return_aux=True)
        scalar_score = float(aux["critic_score"].squeeze().item())
        score = 0.5 * parsed_score + 0.5 * scalar_score
        critic_entropy = float(
            (
                -(
                    F.log_softmax(_logits[:, -1, :].float(), dim=-1).exp()
                    * F.log_softmax(_logits[:, -1, :].float(), dim=-1)
                ).sum(dim=-1)
                / max(math.log(_logits.size(-1)), 1e-8)
            ).item()
        )
        if critic_entropy > 0.75:
            score *= 0.8
        lowered = critique.lower()
        if "uncertain" in lowered or "not sure" in lowered:
            score = max(0.0, score - 0.1)
        return score, critique
    def _run_reasoning_engine(self, user_msg, memory_context=""):
        if not self.reasoning_enabled:
            return None
        base = (
            "You are an internal reasoning engine. Think carefully but concisely.\n"
            f"User request:\n{user_msg}\n\n"
        )
        if memory_context:
            base += f"Relevant memory:\n{memory_context}\n\n"
        planner_prompt = (
            base + "Write a short plan before solving.\nFormat:\nPlan:\n- ...\n"
        )
        plan = self._generate_text(
            planner_prompt,
            max_new_tokens=72,
            temperature=0.3,
            top_k=max(20, min(self.top_k, 40)),
            top_p=0.9,
            min_p=0.0,
            repetition_penalty=1.0,
        )
        plan = self._truncate_generated(
            plan, ["Thought:", "Candidate", "FINAL ANSWER:"]
        )
        best = {"thought": "", "score": -1.0, "critique": ""}
        branches, steps, beam_width = self._reasoning_budget(user_msg)
        seeds = [""]
        for step in range(steps):
            candidates = []
            for seed in seeds:
                for branch in range(branches):
                    thought_prompt = (
                        base + f"{plan}\n\n"
                        "Generate one candidate reasoning path.\n"
                        "If solved, include `FINAL ANSWER:` near the end.\n"
                    )
                    if seed:
                        thought_prompt += (
                            f"\nCurrent best draft:\n{seed}\n\nImprove or fix it.\n"
                        )
                    thought_prompt += f"\nCandidate {branch + 1}:\nThought:\n"
                    thought = self._generate_text(
                        thought_prompt,
                        max_new_tokens=128,
                        temperature=max(0.55, self.temperature),
                        top_k=self.top_k,
                        top_p=self.top_p,
                        min_p=self.min_p,
                        repetition_penalty=self.repetition_penalty,
                    )
                    thought = self._truncate_generated(
                        thought,
                        [f"\nCandidate {branch + 2}:", "\nUser:", "\nAssistant:"],
                    )
                    try:
                        score, critique = self._critic_score(user_msg, plan, thought)
                    except Exception:
                        score, critique = 0.5, "Critic unavailable."
                    candidates.append(
                        {
                            "thought": thought.strip(),
                            "score": score,
                            "critique": critique.strip(),
                        }
                    )
            answer_counts = {}
            for candidate in candidates:
                answer_key = self._extract_final_answer(candidate["thought"]).lower()
                if answer_key:
                    answer_counts[answer_key] = answer_counts.get(answer_key, 0) + 1
            for candidate in candidates:
                answer_key = self._extract_final_answer(candidate["thought"]).lower()
                if answer_key:
                    candidate["score"] += 0.08 * max(
                        0, answer_counts.get(answer_key, 1) - 1
                    )
            candidates.sort(key=lambda item: item["score"], reverse=True)
            step_best = candidates[0]
            if step_best["score"] >= best["score"]:
                best = step_best
            seeds = [item["thought"] for item in candidates[:beam_width]]
            if best["score"] > 0.85:
                break
            if "FINAL ANSWER:" in best["thought"]:
                break
        return {
            "plan": plan.strip(),
            "best_thought": best["thought"].strip(),
            "critique": best["critique"].strip(),
            "score": best["score"],
        }
    def _route(self, msg: str):
        msg_l = msg.lower()
        if re.match(r"^[0-9\+\-\*\/\(\)\.\s]+$", msg_l):
            return "tool"
        if any(k in msg_l for k in ["calculate", "compute"]):
            return "tool"
        if any(k in msg_l for k in ["why", "how", "explain", "compare"]):
            return "reasoning"
        if len(msg.split()) > 25:
            return "reasoning"
        return "fast"
    def respond(self, user_msg, system_prompt=SYSTEM_PROMPT):
        self.model.eval()
        turn_started = time.time()
        mode = self._route(user_msg)
        if mode == "tool":
            res = _safe_tool_eval(user_msg)
            yield res
            return
        elif mode == "reasoning":
            self.set_mode(False)
        else:
            self.set_mode(True)
        max_ctx = self.model.config.block_size
        if len(self.all_token_ids) > max_ctx:
            self.all_token_ids = self.all_token_ids[-max_ctx:]
            # FIX #3: Sync current_len after every all_token_ids update
            self.current_len = len(self.all_token_ids)
            # FIX #5: Reset seen_mask when trimming context (not just on cache reset)
            self._rebuild_seen_mask()
        if len(self.all_token_ids) > int(self.max_cache_len * 0.85):
            self._reset_cache()
            self._rebuild_seen_mask()
            self.global_pos = 0
        if hasattr(self.model, "_reflect_cooldown"):
            self.model._reflect_cooldown = max(0, self.model._reflect_cooldown - 1)
        query_ids = self.tokenizer.encode(user_msg).ids
        if self.fast_mode:
            memory_context = ""
            memory_state = None
            memory_query_state = None
        else:
            try:
                mem_hits = self.memory.retrieve_kv(query_ids)
                if mem_hits and mem_hits[0][0] > 0.6:
                    memory_context = mem_hits[0][2]
                    user_msg = f"[Context]\n{memory_context}\n\n{user_msg}"
                else:
                    memory_context = ""
            except Exception:
                memory_context = ""
            memory_state = None
            memory_query_state = None
        if not self.all_token_ids:
            sys_p = system_prompt
            if memory_context:
                sys_p += f"\n\nRelevant past context:\n{memory_context}"
            messages = build_chat_prompt([], user_msg, sys_p)
        else:
            messages = [{"role": "user", "content": user_msg}]
        prompt_budget = max(
            1, min(self.model.config.block_size - self.max_tokens, self.max_cache_len)
        )
        base_prompt_ids, _ = format_chat(
            messages, self.tokenizer, add_generation_prompt=True
        )
        reasoning_ids = []
        runtime_fast_hint = (
            self.last_telemetry.get("latency_s", 0.0) > 5.0
            or self.last_telemetry.get("tok_s", 999.0) < 8.0
            or self._auto_fast_turns > 0
        )
        force_reasoning = self._force_reasoning_turns > 0
        if (
            self.fast_mode
            or (runtime_fast_hint and not force_reasoning)
            or len(user_msg) < 100
        ):
            reasoning_bundle = None
        else:
            try:
                reasoning_bundle = self._run_reasoning_engine(
                    user_msg, memory_context=memory_context
                )
            except Exception:
                reasoning_bundle = None
        if reasoning_bundle:
            hidden_reason = (
                "\nPlan:\n"
                f"{reasoning_bundle['plan']}\n\n"
                "Best Thought:\n"
                f"{reasoning_bundle['best_thought']}\n\n"
                "Critic:\n"
                f"{reasoning_bundle['critique']}\n"
                f"Critic Score: {reasoning_bundle['score']:.2f}\n\n"
                "Use tools if necessary. You may chain tools before answering.\n"
                "Do not reveal the plan, thought graph, critique, or internal notes.\n"
                "Then provide only the final user-facing response.\n"
                "FINAL ANSWER:\n"
            )
            raw_reasoning_ids = self.tokenizer.encode(hidden_reason).ids
            reasoning_ids = self._cap_hidden_suffix(
                base_prompt_ids, raw_reasoning_ids, prompt_budget, MAX_REASONING_TOKENS
            )
        new_ids = base_prompt_ids + reasoning_ids
        total_len = len(self.all_token_ids) + len(new_ids)
        if total_len > prompt_budget:
            self._reset_cache()
            self.global_pos = 0
            self.seen_mask.zero_()
            while self.history:
                messages = build_chat_prompt(self.history, user_msg, system_prompt)
                base_prompt_ids, _ = format_chat(
                    messages, self.tokenizer, add_generation_prompt=True
                )
                reasoning_ids = self._cap_hidden_suffix(
                    base_prompt_ids, reasoning_ids, prompt_budget, MAX_REASONING_TOKENS
                )
                new_ids = base_prompt_ids + reasoning_ids
                if len(new_ids) <= prompt_budget:
                    break
                self.history = self.history[1:]
                self._rebuild_seen_mask()
            self._rebuild_seen_mask()
            max_prompt = prompt_budget
            base_prompt_ids, _ = format_chat(
                messages, self.tokenizer, add_generation_prompt=True
            )
            reasoning_ids = self._cap_hidden_suffix(
                base_prompt_ids, reasoning_ids, max_prompt, MAX_REASONING_TOKENS
            )
            room_for_prompt = max(1, max_prompt - len(reasoning_ids))
            new_ids = base_prompt_ids[-room_for_prompt:] + reasoning_ids
        eos_id = self.tokenizer.token_to_id(EOS_TOKEN)
        usr_id = self.tokenizer.token_to_id(USR_TOKEN)
        amp_dtype = (
            next(self.model.parameters()).dtype
            if self.device == "cuda"
            else torch.float32
        )
        amp_ctx = make_amp_context(self.device, amp_dtype)
        with torch.no_grad(), amp_ctx:
            new_t = torch.tensor([new_ids], dtype=torch.long, device=self.device)
            x = self.model.transformer.wte(new_t)
            x = self._inject_memory_state(x, memory_state, memory_query_state)
            start_pos = self.global_pos
            freqs_cos, freqs_sin = self._rope_slice(
                start_pos, len(new_ids), device=x.device
            )
            for i, block in enumerate(self.model.transformer.h):
                x, self.caches[i] = block(
                    x, freqs_cos, freqs_sin, kv_cache=self.caches[i]
                )
            x = self.model.transformer.ln_f(x)
            logits = self.model.lm_head(x[:, -1, :])
            self.all_so_far[0, self.current_len : self.current_len + len(new_ids)] = (
                new_t[0]
            )
            self.current_len += len(new_ids)
            self.global_pos += len(new_ids)
            self.all_token_ids.extend(new_ids)
            # FIX #3: Sync current_len after every all_token_ids update
            self.current_len = len(self.all_token_ids)
            if len(self.all_token_ids) > self.model.config.block_size:
                self.all_token_ids = self.all_token_ids[-self.model.config.block_size:]
                self.current_len = len(self.all_token_ids)
            self._mark_seen(new_ids)
            self._assert_state_sync()
            generated_ids = []
            self._has_reflected = False
            self._tool_calls = 0
            self._processed_tool_calls = 0
            hard_stop = False
            idx_next = self.model._sample_token(
                logits,
                self.all_so_far[:, : self.current_len],
                self.temperature,
                self.top_k,
                self.top_p,
                self.min_p,
                self.repetition_penalty,
                seen_mask=self.seen_mask,
            )
            for _ in range(self.max_tokens):
                token_id = idx_next.item()
                if token_id == eos_id or token_id == usr_id:
                    hard_markers = ("why", "prove", "analyze", "debug", "design")
                    if (
                        (not self.fast_mode)
                        and reasoning_bundle is None
                        and len(user_msg) >= 80
                        and any(m in user_msg.lower() for m in hard_markers)
                        and self.reflection_enabled
                        and not self._has_reflected
                        and self.model._should_reflect(self.temperature)
                    ):
                        self._has_reflected = True  
                        self.model._reflect_cooldown = max(
                            getattr(self.model, "_reflect_cooldown", 0), 2
                        )
                        reflect_text = "\n<|user|>\nIs the above reasoning correct? Please verify, fix any mistakes, and provide the final answer.\n<|assistant|>\nThought:\n"
                        reflect_ids = self._cap_runtime_injection(
                            self.tokenizer.encode(reflect_text).ids,
                            MAX_REFLECTION_TOKENS,
                            reserve_tokens=1,
                        )
                        if not reflect_ids:
                            break
                        self.all_token_ids.extend(reflect_ids)
                        # FIX #3: Sync current_len after every all_token_ids update
                        self.current_len = len(self.all_token_ids)
                        new_t = torch.tensor(
                            [reflect_ids], dtype=torch.long, device=self.device
                        )
                        x_ref = self.model.transformer.wte(new_t)
                        x_ref = self._inject_memory_state(
                            x_ref, memory_state, memory_query_state
                        )
                        self.all_so_far[
                            0, self.current_len : self.current_len + len(reflect_ids)
                        ] = new_t[0]
                        self.current_len += len(reflect_ids)
                        fc, fs = self._rope_slice(self.global_pos, len(reflect_ids))
                        self.global_pos += len(reflect_ids)
                        for i, block in enumerate(self.model.transformer.h):
                            x_ref, self.caches[i] = block(
                                x_ref, fc, fs, kv_cache=self.caches[i]
                            )
                        x_ref = self.model.transformer.ln_f(x_ref)
                        new_logits = self.model.lm_head(x_ref[:, -1, :])
                        logits = new_logits
                        idx_next = self.model._sample_token(
                            new_logits,
                            self.all_so_far[:, : self.current_len],
                            self.temperature,
                            self.top_k,
                            self.top_p,
                            self.min_p,
                            self.repetition_penalty,
                            seen_mask=self.seen_mask,
                        )
                        for cache in self.caches:
                            cache.rollback(len(reflect_ids))
                        self.current_len -= len(reflect_ids)
                        self.global_pos -= len(reflect_ids)
                        del self.all_token_ids[-len(reflect_ids) :]
                        # FIX #3: current_len already updated, verify sync
                        assert self.current_len == len(self.all_token_ids), "current_len desync after rollback"
                        self.all_so_far[
                            0, self.current_len : self.current_len + len(reflect_ids)
                        ] = 0
                        self._rebuild_seen_mask()
                        self._assert_state_sync()
                        continue
                    else:
                        break
                generated_ids.append(token_id)
                self.all_token_ids.append(token_id)
                if len(self.all_token_ids) > self.model.config.block_size:
                    self.all_token_ids = self.all_token_ids[-self.model.config.block_size:]
                # FIX #3: Sync current_len after every all_token_ids update
                self.current_len = len(self.all_token_ids)
                self.all_so_far[0, self.current_len - 1] = token_id
                self.global_pos += 1
                self._mark_seen([token_id])
                self._assert_state_sync()
                current_text = self.tokenizer.decode(generated_ids)
                visible_text = self._visible_response(current_text)
                # NEXA 1.3: Stream every token for real-time UX (not every 4)
                yield visible_text
                if not user_msg.lower().startswith("calc:"):
                    calls = 0
                else:
                    calls = current_text.count("CALL: python(")
                if calls >= self.max_tool_chain:
                    yield visible_text
                    return
                if calls > getattr(self, "_processed_tool_calls", 0):
                    start_idx = current_text.rfind("CALL: python(") + 13
                    end_idx = _find_balanced_call_end(current_text, start_idx)
                    if end_idx == -1:
                        if len(current_text) - start_idx > 250:
                            self._processed_tool_calls = (
                                getattr(self, "_processed_tool_calls", 0) + 1
                            )
                            yield (
                                visible_text
                                + "\n[System Log: Unterminated tool call blocked]\n"
                            )
                            return
                        continue
                    expr = current_text[start_idx:end_idx].strip()
                    self._processed_tool_calls = (
                        getattr(self, "_processed_tool_calls", 0) + 1
                    )
                    self._tool_calls += 1
                    time.sleep(0.1)  
                    if self._tool_calls >= self.max_tool_chain:
                        yield visible_text
                        return  
                    if len(expr) > 250:
                        res = "Error: unsafe expression or blocked tokens"
                    else:
                        try:
                            fut = self._ensure_executor().submit(_safe_tool_eval, expr)
                            res = fut.result(timeout=2.0)[:1000]
                        except FuturesTimeout:
                            fut.cancel()
                            # NEXA 1.2: Cleanup and recreate executor to kill zombie process
                            self._cleanup_executor()
                            self._ensure_executor()
                            res = "Error: timeout (tool hung)"
                        except Exception as e:
                            res = f"Error: {e}"
                    if res.startswith("Error:"):
                        self._log_fail_case(
                            user_msg,
                            current_text,
                            "tool_failure",
                            {"expr": expr, "result": res},
                        )
                    tool_memory_text = f"Tool expression: {expr}\nTool result: {res}"
                    self.memory.add(
                        tool_memory_text,
                        self.tokenizer.encode(tool_memory_text).ids,
                        memory_type="tool_result",
                    )
                    feed_text = (
                        f"\n<|system|>\nTool result: {res}\n<|assistant|>\nAnswer:\n"
                    )
                    feed_ids = self._cap_runtime_injection(
                        self.tokenizer.encode(feed_text).ids,
                        hard_cap=min(192, self.max_cache_len),
                        reserve_tokens=1,
                    )
                    if not feed_ids:
                        yield (
                            visible_text
                            + "\n[System Log: Tool result skipped due to context budget]\n"
                        )
                        return
                    self.all_token_ids.extend(feed_ids)
                    # FIX #3: Sync current_len after every all_token_ids update
                    self.current_len = len(self.all_token_ids)
                    yield (
                        visible_text
                        + f"\n[System Log: Tool result {res}]\n"
                    )
                    new_t = torch.tensor(
                        [feed_ids], dtype=torch.long, device=self.device
                    )
                    x_feed = self.model.transformer.wte(new_t)
                    x_feed = self._inject_memory_state(
                        x_feed, memory_state, memory_query_state
                    )
                    self.all_so_far[
                        0, self.current_len : self.current_len + len(feed_ids)
                    ] = new_t[0]
                    self.current_len += len(feed_ids)
                    fc, fs = self._rope_slice(self.global_pos, len(feed_ids))
                    self.global_pos += len(feed_ids)
                    for i, block in enumerate(self.model.transformer.h):
                        x_feed, self.caches[i] = block(
                            x_feed, fc, fs, kv_cache=self.caches[i]
                        )
                    x_feed = self.model.transformer.ln_f(x_feed)
                    new_logits = self.model.lm_head(x_feed[:, -1, :])
                    logits = new_logits
                    idx_next = self.model._sample_token(
                        new_logits,
                        self.all_so_far[:, : self.current_len],
                        self.temperature,
                        self.top_k,
                        self.top_p,
                        self.min_p,
                        self.repetition_penalty,
                        seen_mask=self.seen_mask,
                    )
                    self._mark_seen(feed_ids)
                    self._assert_state_sync()
                    continue
                if self.current_len >= self.max_cache_len:
                    self._reset_cache()
                    if self.history:
                        self.history = self.history[1:]
                    self._rebuild_seen_mask()
                    messages = build_chat_prompt(self.history, user_msg, system_prompt)
                    ctx_ids, _ = format_chat(
                        messages, self.tokenizer, add_generation_prompt=True
                    )
                    ctx_ids.extend(generated_ids)
                    max_ctx = self.model.config.block_size - self.max_tokens
                    if len(ctx_ids) > max_ctx:
                        ctx_ids = ctx_ids[-max_ctx:]
                    self.all_token_ids = ctx_ids.copy()
                    # FIX #3: Sync current_len after every all_token_ids update
                    self.current_len = len(self.all_token_ids)
                    self.seen_mask.zero_()
                    self._mark_seen(ctx_ids)
                    ctx_t = torch.tensor(
                        [ctx_ids], dtype=torch.long, device=self.device
                    )
                    x_ctx = self.model.transformer.wte(ctx_t)
                    x_ctx = self._inject_memory_state(
                        x_ctx, memory_state, memory_query_state
                    )
                    self.current_len = len(ctx_ids)
                    self.global_pos = 0
                    fc, fs = self._rope_slice(0, self.current_len, device=x_ctx.device)
                    for i, block in enumerate(self.model.transformer.h):
                        x_ctx, self.caches[i] = block(
                            x_ctx, fc, fs, kv_cache=self.caches[i]
                        )
                    x_ctx = self.model.transformer.ln_f(x_ctx)
                    new_logits = self.model.lm_head(x_ctx[:, -1, :])
                    self.all_so_far[0, : self.current_len] = ctx_t[0]
                    self.global_pos = self.current_len
                    logits = new_logits
                    self._assert_state_sync()
                    idx_next = self.model._sample_token(
                        new_logits,
                        self.all_so_far[:, : self.current_len],
                        self.temperature,
                        self.top_k,
                        self.top_p,
                        self.min_p,
                        self.repetition_penalty,
                        seen_mask=self.seen_mask,
                    )
                    continue
                x = self.model.transformer.wte(idx_next)
                x = self._inject_memory_state(x, memory_state, memory_query_state)
                freqs_cos, freqs_sin = self._rope_slice(self.global_pos - 1, 1)
                for i, block in enumerate(self.model.transformer.h):
                    x, self.caches[i] = block(
                        x, freqs_cos, freqs_sin, kv_cache=self.caches[i]
                    )
                x = self.model.transformer.ln_f(x)
                logits = self.model.lm_head(x[:, -1, :])
                idx_next = self.model._sample_token(
                    logits,
                    self.all_so_far[:, : self.current_len],
                    self.temperature,
                    self.top_k,
                    self.top_p,
                    self.min_p,
                    self.repetition_penalty,
                    seen_mask=self.seen_mask,
                )
                if hard_stop:
                    break
        max_ctx = self.model.config.block_size
        if len(self.all_token_ids) > max_ctx:
            self.all_token_ids = self.all_token_ids[-max_ctx:]
            # FIX #3: Sync current_len after every all_token_ids update
            self.current_len = len(self.all_token_ids)
        response = self._visible_response(
            self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        )
        self.history.append((user_msg, response))
        self.turn_count += 1
        turn_text = f"User: {user_msg}\nAssistant: {response}"
        turn_ids = self.tokenizer.encode(turn_text).ids
        self.memory.add(turn_text, turn_ids, memory_type="conversation")
        if response.strip():
            fact_text = f"Assistant fact:\n{response}"
            self.memory.add(
                fact_text, self.tokenizer.encode(fact_text).ids, memory_type="fact"
            )
        self._has_reflected = False  
        self._tool_calls = 0
        if len(self.history) % 20 == 0:
            with self.memory.lock:
                self.memory.db.clear()
                # NEXA 1.3: Invalidate cache on clear
                self.memory._emb_cache_dirty = True
                self.memory._emb_cache_tensor = None
        elapsed = max(time.time() - turn_started, 1e-6)
        tok_count = len(generated_ids)
        self.last_telemetry = {
            "latency_s": elapsed,
            "tokens": tok_count,
            "tok_s": tok_count / elapsed if tok_count > 0 else 0.0,
            "critic_score": (reasoning_bundle or {}).get("score"),
            "spec_accept_ema": getattr(self.model, "_spec_accept_ema", None),
            "spec_gamma": getattr(self.model, "_adaptive_gamma", None),
            "spec_disabled_steps": getattr(self.model, "_spec_disable_steps", None),
        }
        if self.last_telemetry["latency_s"] > 5.0 or self.last_telemetry["tok_s"] < 8.0:
            self._auto_fast_turns = 2
            self._good_runtime_streak = 0
        else:
            self._good_runtime_streak += 1
            if self._good_runtime_streak >= 2:
                self._auto_fast_turns = max(0, self._auto_fast_turns - 1)
        if (
            self.last_telemetry["critic_score"] is not None
            and self.last_telemetry["critic_score"] < 0.5
        ):
            self._force_reasoning_turns = 2
        else:
            self._force_reasoning_turns = max(0, self._force_reasoning_turns - 1)
        if (
            self.last_telemetry["critic_score"] is not None
            and self.last_telemetry["critic_score"] < 0.35
        ):
            self._log_fail_case(
                user_msg,
                response,
                "low_critic_score",
                {"critic_score": self.last_telemetry["critic_score"]},
            )
        print(
            f"[telemetry] latency={self.last_telemetry['latency_s']:.2f}s "
            f"tok/s={self.last_telemetry['tok_s']:.1f} "
            f"tokens={tok_count} "
            f"critic={self.last_telemetry['critic_score']} "
            f"spec_accept={self.last_telemetry['spec_accept_ema']} "
            f"spec_gamma={self.last_telemetry['spec_gamma']} "
            f"spec_disabled={self.last_telemetry['spec_disabled_steps']}"
        )
        if len(self.history) > 20:
            self.history = self.history[-20:]
    def reset(self):
        self.history = []
        self._reset_cache()
        with self.memory.lock:
            self.memory.db.clear()
            # NEXA 1.3: Invalidate embedding cache on reset
            self.memory._emb_cache_dirty = True
            self.memory._emb_cache_tensor = None
        self.global_pos = 0
        if hasattr(self.model, "seen_mask") and self.model.seen_mask is not None:
            self.model.seen_mask.zero_()
        if hasattr(self.model, "_reflect_cooldown"):
            self.model._reflect_cooldown = 0
        # FIX #6: Cleanup executor on reset
        self._cleanup_executor()

    def close(self):
        # FIX #7: Properly shutdown executor to prevent zombie processes
        self._cleanup_executor()
    def __del__(self):
        try:
            # FIX #1: Full lifecycle cleanup - executor first, then close
            self._cleanup_executor()
            self.close()
        except Exception:
            pass
def sft_finetune(
    model,
    tokenizer,
    data_path,
    config,
    lora_rank=8,
    lora_alpha=16.0,
    epochs=3,
    lr=2e-4,
    save_dir="lora_checkpoints",
):
    device = config.device
    model = apply_lora(model, rank=lora_rank, alpha=lora_alpha)
    model.train()
    model.transformer.wte.weight.requires_grad_(True)
    print(f"Loading SFT data: {data_path}...")
    if os.path.exists(data_path):
        with open(data_path) as f:
            data = [json.loads(line) for line in f if line.strip()]
    else:
        print("Downloading from Hugging Face Hub...")
        from datasets import load_dataset
        try:
            ds = load_dataset(data_path, split="train")
        except ValueError:
            ds = load_dataset(data_path, split="train_sft")
        data = [item for item in ds]
    if os.path.exists(SELF_IMPROVE_DATASET_PATH):
        with open(SELF_IMPROVE_DATASET_PATH, encoding="utf-8") as f:
            data.extend(json.loads(line) for line in f if line.strip())
    print(f"  {len(data)} examples")
    reasoning_bank = []
    for item in data:
        for msg in item.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            thought_text, answer_text = _split_reasoning_answer(msg.get("content", ""))
            if answer_text:
                reasoning_bank.append((thought_text, answer_text))
    print("Tokenizing with loss masking...")
    examples = []
    rank_pairs = []
    memory_bank = []
    for idx_item, item in enumerate(data):
        ids, mask, reasoning_mask = format_chat(
            item["messages"],
            tokenizer,
            add_generation_prompt=False,
            return_reasoning_mask=True,
        )
        if len(ids) <= config.block_size:
            critic_label = item.get(
                "critic_label", 1.0 if any(reasoning_mask) else None
            )
            examples.append((ids, mask, reasoning_mask, critic_label))
            memory_bank.append(ids)
            pairwise = item.get("pairwise")
            if pairwise and pairwise.get("chosen") and pairwise.get("rejected"):
                chosen_ids = tokenizer.encode(pairwise["chosen"]).ids[
                    : config.block_size
                ]
                rejected_ids = tokenizer.encode(pairwise["rejected"]).ids[
                    : config.block_size
                ]
                if chosen_ids and rejected_ids:
                    rank_pairs.append((chosen_ids, rejected_ids))
        trace_messages = _build_training_trace_messages(item["messages"])
        trace_ids, trace_mask, trace_reasoning_mask = format_chat(
            trace_messages,
            tokenizer,
            add_generation_prompt=False,
            return_reasoning_mask=True,
        )
        if len(trace_ids) <= config.block_size:
            examples.append(
                (
                    trace_ids,
                    trace_mask,
                    trace_reasoning_mask,
                    1.0 if any(trace_reasoning_mask) else None,
                )
            )
        if not reasoning_bank:
            continue
        neg_messages = None
        for msg in item.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            thought_text, answer_text = _split_reasoning_answer(msg.get("content", ""))
            if not answer_text:
                continue
            wrong_thought = None
            for bank_idx, (cand_thought, cand_answer) in enumerate(reasoning_bank):
                if cand_answer != answer_text and cand_thought.strip():
                    wrong_thought = cand_thought
                    break
            if wrong_thought is None:
                wrong_thought = "The reasoning does not support the final answer."
            neg_messages = []
            for cur_msg in item["messages"]:
                if cur_msg is msg:
                    neg_messages.append(
                        {
                            "role": "assistant",
                            "content": _build_reasoning_scaffold(
                                wrong_thought,
                                answer_text,
                                score_text="0.0",
                                refine_text="This reasoning is inconsistent and should not be trusted.",
                            ),
                        }
                    )
                else:
                    neg_messages.append(cur_msg)
            break
        if neg_messages is not None:
            neg_ids, _neg_mask, neg_reasoning_mask = format_chat(
                neg_messages,
                tokenizer,
                add_generation_prompt=False,
                return_reasoning_mask=True,
            )
            if len(neg_ids) <= config.block_size:
                examples.append(
                    (neg_ids, [False] * len(neg_ids), [False] * len(neg_ids), 0.0)
                )
                if len(ids) <= config.block_size:
                    rank_pairs.append((ids, neg_ids))
    print(f"  {len(examples)} fit in block_size={config.block_size}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.0,
    )
    use_amp = device == "cuda"
    amp_dtype = next(model.parameters()).dtype if use_amp else torch.float32
    amp_ctx = make_amp_context(device, amp_dtype)
    scaler = torch.amp.GradScaler(device) if use_amp else None
    print(f"\nSFT: {epochs} epochs, lr={lr}")
    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(epochs):
        random.shuffle(examples)
        total_loss, n_batches = 0, 0
        for ids, mask, reasoning_mask, critic_label in examples:
            pad_id = getattr(config, "eos_id", 0)
            try:
                pad_id = tokenizer.token_to_id(PAD_TOKEN)
            except Exception:
                pass
            pad_len = config.block_size - len(ids)
            padded_ids = ids + [pad_id] * (pad_len + 1)
            padded_mask = mask + [False] * (pad_len + 1)
            padded_reasoning_mask = reasoning_mask + [False] * (pad_len + 1)
            x = torch.tensor([padded_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([padded_ids[1:]], dtype=torch.long, device=device)
            memory_state = None
            memory_query_state = None
            memory_mode = "none"
            mem_rand = random.random()
            if mem_rand >= getattr(config, "memory_train_dropout", 0.3):
                with torch.no_grad():
                    if mem_rand < 0.6 and memory_bank:
                        wrong_ids = random.choice(memory_bank)
                        mem_x = torch.tensor(
                            [wrong_ids[: config.block_size]],
                            dtype=torch.long,
                            device=device,
                        )
                        tok = model.transformer.wte(mem_x).detach()
                        memory_mode = "wrong"
                    else:
                        tok = model.transformer.wte(x).detach()
                        memory_mode = "true"
                    memory_state = F.normalize(tok.mean(dim=1), dim=-1)
                    memory_query_state = memory_state.clone()
            loss_mask = torch.tensor([padded_mask[1:]], dtype=torch.bool, device=device)
            reasoning_loss_mask = torch.tensor(
                [padded_reasoning_mask[1:]], dtype=torch.bool, device=device
            )
            if loss_mask.sum() == 0 and critic_label is None:
                continue
            if loss_mask.sum() > 0:
                y[~loss_mask] = -100
            else:
                y = None
            reasoning_targets = torch.tensor(
                [padded_ids[1:]], dtype=torch.long, device=device
            )
            reasoning_targets[~reasoning_loss_mask] = -100
            critic_labels = None
            if critic_label is not None:
                critic_labels = torch.full(
                    (1, 1), float(critic_label), dtype=torch.float32, device=device
                )
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                _logits, loss, aux = model(
                    x,
                    y,
                    reasoning_targets=reasoning_targets
                    if (y is not None and reasoning_loss_mask.any())
                    else None,
                    critic_labels=critic_labels,
                    memory_state=memory_state,
                    memory_query_state=memory_query_state,
                    return_aux=True,
                )
                if rank_pairs:
                    pos_ids, neg_ids = random.choice(rank_pairs)
                    pos_x = torch.tensor(
                        [pos_ids[: config.block_size]], dtype=torch.long, device=device
                    )
                    neg_x = torch.tensor(
                        [neg_ids[: config.block_size]], dtype=torch.long, device=device
                    )
                    _p_logits, _p_loss, pos_aux = model(pos_x, return_aux=True)
                    _n_logits, _n_loss, neg_aux = model(neg_x, return_aux=True)
                    rank_loss = F.margin_ranking_loss(
                        pos_aux["critic_score"].view(-1),
                        neg_aux["critic_score"].view(-1),
                        torch.ones(1, device=device),
                        margin=0.25,
                    )
                    rank_weight = getattr(config, "critic_rank_loss_weight", 1.0)
                    loss = (
                        rank_loss * rank_weight
                        if loss is None
                        else loss + rank_weight * rank_loss
                    )
                if memory_mode != "none" and critic_labels is not None:
                    with torch.no_grad():
                        _nm_logits, _nm_loss, nm_aux = model(
                            x,
                            y,
                            critic_labels=critic_labels,
                            return_aux=True,
                        )
                    mem_delta = aux["critic_score"] - nm_aux["critic_score"]
                    target = (
                        torch.ones_like(mem_delta)
                        if memory_mode == "true"
                        else torch.zeros_like(mem_delta)
                    )
                    usefulness_loss = F.binary_cross_entropy(
                        torch.sigmoid(mem_delta * 5.0), target
                    )
                    loss = loss + 0.02 * usefulness_loss
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(
            f"  epoch {epoch + 1}/{epochs} | loss {avg:.4f} | {time.time() - t0:.0f}s"
        )
        if avg < best_loss:
            best_loss = avg
            save_lora(model, os.path.join(save_dir, "best.pt"))
    print(f"\nSFT Done! Best loss: {best_loss:.4f}")
    return model
def build_self_improve_dataset(
    model,
    tokenizer,
    device,
    fail_log_path=FAIL_LOG_PATH,
    out_path=SELF_IMPROVE_DATASET_PATH,
    max_cases=200,
    critic_model=None,
):
    if not os.path.exists(fail_log_path):
        print(f"[warn] No fail log found at {fail_log_path}")
        return 0
    with open(fail_log_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = rows[-max_cases:]
    session = ChatSession(
        model, tokenizer, device, fast_mode=False, critic_model=critic_model
    )
    built = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out_f:
        fail_rows = rows[: max_cases // 2]
        normal_rows = [
            {"user": q, "reason": "normal_mix", "response": "", "extra": {}}
            for q in NORMAL_QUERY_SEEDS[: max(1, len(fail_rows))]
        ]
        while len(normal_rows) < len(fail_rows):
            normal_rows.extend(
                {"user": q, "reason": "normal_mix", "response": "", "extra": {}}
                for q in NORMAL_QUERY_SEEDS
            )
            normal_rows = normal_rows[: len(fail_rows)]
        selected_rows = fail_rows + normal_rows
        for row in selected_rows:
            user_msg = row.get("user", "").strip()
            if not user_msg:
                continue
            memory_context = ""
            prompt = session._build_answer_prompt(
                user_msg, memory_context=memory_context
            )
            answers = []
            for temp in (0.4, 0.7, 1.0):
                text = session._generate_text(
                    prompt,
                    max_new_tokens=min(160, session.max_tokens),
                    temperature=temp,
                    top_k=session.top_k,
                    top_p=session.top_p,
                    min_p=session.min_p,
                    repetition_penalty=session.repetition_penalty,
                )
                answers.append(session._visible_response(text).strip())
            uniq_answers = []
            for ans in answers:
                if not ans:
                    continue
                if any(
                    SequenceMatcher(None, ans, prev).ratio() > 0.8
                    for prev in uniq_answers
                ):
                    continue
                uniq_answers.append(ans)
            answers = uniq_answers
            if len(answers) < 2:
                continue
            scored = []
            for ans in answers:
                score, critique = session._critic_score(
                    user_msg, "Answer directly.", f"FINAL ANSWER: {ans}"
                )
                score = max(0.0, min(1.0, score + random.uniform(-0.05, 0.05)))
                scored.append((score, ans, critique))
            scored.sort(key=lambda x: x[0], reverse=True)
            best_score, best_ans, best_critique = scored[0]
            worst_score, worst_ans, worst_critique = scored[-1]
            if abs(best_score - worst_score) < 0.1:
                continue
            auto_label = None
            if best_score > 0.8:
                auto_label = 1.0
            elif best_score < 0.4:
                auto_label = 0.0
            if auto_label is not None:
                out_f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": user_msg},
                                {
                                    "role": "assistant",
                                    "content": _build_reasoning_scaffold(
                                        best_critique or "Reasoning accepted.",
                                        best_ans,
                                        score_text=f"{best_score:.2f}",
                                        refine_text=best_ans,
                                    ),
                                },
                            ],
                            "critic_label": auto_label,
                            "source": "self_improve_auto_label",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                built += 1
            if best_ans and worst_ans and best_ans != worst_ans:
                out_f.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "user", "content": user_msg},
                                {
                                    "role": "assistant",
                                    "content": _build_reasoning_scaffold(
                                        best_critique or "Preferred answer.",
                                        best_ans,
                                        score_text=f"{best_score:.2f}",
                                        refine_text=best_ans,
                                    ),
                                },
                            ],
                            "pairwise": {
                                "chosen": best_ans,
                                "rejected": worst_ans,
                                "chosen_score": best_score,
                                "rejected_score": worst_score,
                            },
                            "source": "self_improve_pairwise",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                built += 1
    session.close()
    print(f"[self-improve] wrote {built} examples to {out_path}")
    return built
def launch_openai_api(
    model, tokenizer, device, critic_model=None, port=7860
):
    """
    NEXA 1.3: Production-grade OpenAI-compatible API server.

    Features:
    - FastAPI + uvicorn for async performance
    - True SSE streaming (token-by-token)
    - Multi-session isolation (per user/session)
    - Dynamic batching scheduler
    - CORS support
    - Proper error handling
    - Usage stats in response
    - /v1/chat/completions (POST, stream + non-stream)
    - /v1/models (GET)
    - /v1/sessions (GET - list active sessions)

    Falls back to stdlib http.server if FastAPI/uvicorn not installed.
    """
    # Try to compile model for faster inference
    if hasattr(torch, "compile"):
        try:
            torch._dynamo.config.cache_size_limit = 64
            model.forward = torch.compile(
                model.forward, mode="reduce-overhead", fullgraph=False
            )
            print("Model forward compiled for inference!")
        except Exception as e:
            print(f"[warn] Compile failed for API: {e}")

    # Try FastAPI (production)
    try:
        from fastapi import FastAPI, Request, HTTPException
        from fastapi.responses import StreamingResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        import uvicorn
        _has_fastapi = True
    except ImportError:
        _has_fastapi = False
        print("[warn] FastAPI/uvicorn not installed. Falling back to stdlib server.")
        print("       Install with: pip install fastapi uvicorn")

    # Create session manager
    session_mgr = SessionManager(
        model, tokenizer, device,
        critic_model=critic_model,
        max_sessions=50,
        session_ttl=1800,
        fast_mode=False,
    )

    # Create batch scheduler
    batch_scheduler = DynamicBatchScheduler(
        model, tokenizer, device,
        max_batch_size=8,
        batch_timeout_ms=50,
        max_seq_len=model.config.block_size,
    )
    batch_scheduler.start()

    if _has_fastapi:
        app = FastAPI(title="Nexa API", version=NEXA_VERSION)

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": "nexa-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "nexa",
                        "permission": [],
                    }
                ],
            }

        @app.get("/v1/sessions")
        async def list_sessions():
            """NEXA 1.3: List active sessions for monitoring."""
            return {
                "active_sessions": session_mgr.active_count,
                "sessions": session_mgr.list_sessions(),
                "batch_stats": batch_scheduler.stats,
            }

        @app.delete("/v1/sessions/{session_id}")
        async def delete_session(session_id: str):
            """NEXA 1.3: Delete a specific session."""
            session_mgr.remove(session_id)
            return {"status": "deleted", "session_id": session_id}

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid JSON body")

            messages = body.get("messages", [])
            if not messages:
                raise HTTPException(status_code=400, detail="messages is required")

            user_msg = ""
            system_prompt = SYSTEM_PROMPT
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_msg = content

            if not user_msg:
                user_msg = messages[-1].get("content", "")

            stream = body.get("stream", False)
            model_name = body.get("model", "nexa-model")
            req_temperature = body.get("temperature")
            req_max_tokens = body.get("max_tokens")

            # NEXA 1.3: Session isolation per user
            session_id = body.get("session_id", "")
            if not session_id:
                # Try to derive from auth header or use default
                auth = request.headers.get("authorization", "")
                session_id = auth.replace("Bearer ", "").strip() or "default"

            session = session_mgr.get_or_create(session_id)

            # Apply per-request overrides
            if req_temperature is not None:
                session.temperature = float(req_temperature)
            if req_max_tokens is not None:
                session.max_tokens = int(req_max_tokens)

            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            if stream:
                async def stream_response():
                    # Role chunk first
                    role_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant"},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(role_chunk)}\n\n"

                    partial_len = 0
                    total_tokens = 0
                    token_queue = queue.Queue()

                    def _sync_generate():
                        """Run sync generator in thread to avoid blocking event loop."""
                        try:
                            for partial in session.respond(user_msg, system_prompt=system_prompt):
                                token_queue.put(("token", partial))
                            token_queue.put(("done", None))
                        except Exception as e:
                            token_queue.put(("error", str(e)))

                    # Start generation in background thread
                    gen_thread = threading.Thread(target=_sync_generate, daemon=True)
                    gen_thread.start()

                    try:
                        while True:
                            # Non-blocking poll to keep event loop responsive
                            try:
                                msg_type, msg_data = token_queue.get(timeout=0.05)
                            except queue.Empty:
                                await asyncio.sleep(0.01)
                                continue

                            if msg_type == "done":
                                break
                            elif msg_type == "error":
                                error_chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": f"\n[Error: {msg_data}]"},
                                            "finish_reason": "error",
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(error_chunk)}\n\n"
                                break
                            elif msg_type == "token":
                                delta = msg_data[partial_len:]
                                partial_len = len(msg_data)
                                if not delta:
                                    continue
                                total_tokens += 1
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": delta},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                    except Exception as e:
                        error_chunk = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": f"\n[Error: {e}]"},
                                    "finish_reason": "error",
                                }
                            ],
                        }
                        yield f"data: {json.dumps(error_chunk)}\n\n"

                    # Wait for thread to finish
                    gen_thread.join(timeout=5.0)

                    # Final chunk
                    done_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "completion_tokens": total_tokens,
                            "total_tokens": total_tokens,
                        },
                    }
                    yield f"data: {json.dumps(done_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                # Non-streaming
                full_resp = ""
                token_count = 0
                try:
                    for partial in session.respond(user_msg, system_prompt=system_prompt):
                        full_resp = partial
                        token_count += 1
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

                return JSONResponse({
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": full_resp,
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(tokenizer.encode(user_msg).ids),
                        "completion_tokens": token_count,
                        "total_tokens": len(tokenizer.encode(user_msg).ids) + token_count,
                    },
                })

        @app.on_event("shutdown")
        async def shutdown_event():
            batch_scheduler.stop()
            session_mgr.close_all()

        print(f"\n" + "=" * 50)
        print(f"  Nexa {NEXA_VERSION} API Server (FastAPI)")
        print(f"  http://0.0.0.0:{port}/v1")
        print(f"  Endpoints:")
        print(f"    POST /v1/chat/completions")
        print(f"    GET  /v1/models")
        print(f"    GET  /v1/sessions")
        print(f"  Features:")
        print(f"    Dynamic Batching (max_batch={batch_scheduler.max_batch_size})")
        print(f"    Multi-Session (max={session_mgr.max_sessions}, ttl={session_mgr.session_ttl}s)")
        print(f"    Token-by-Token Streaming")
        print("=" * 50 + "\n")

        try:
            uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
        except KeyboardInterrupt:
            print("\nShutting down API server...")
            batch_scheduler.stop()
            session_mgr.close_all()
    else:
        # Fallback: stdlib http.server (legacy)
        import http.server
        import socketserver

        session = session_mgr.get_or_create("default")

        class OpenAIHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass
            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
                self.end_headers()
            def end_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                super().end_headers()
            def do_GET(self):
                if self.path == '/v1/models':
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    res = {
                        "object": "list",
                        "data": [{"id": "nexa-model", "object": "model", "created": int(time.time()), "owned_by": "nexa"}]
                    }
                    self.wfile.write(json.dumps(res).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
            def do_POST(self):
                if self.path == '/v1/chat/completions':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    req = json.loads(post_data.decode('utf-8'))
                    messages = req.get("messages", [])
                    user_msg = messages[-1]["content"] if messages else ""
                    stream = req.get("stream", False)

                    # NEXA 1.3: Session isolation via header
                    auth = self.headers.get("Authorization", "")
                    sid = req.get("session_id", "") or auth.replace("Bearer ", "").strip() or "default"
                    sess = session_mgr.get_or_create(sid)

                    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

                    if stream:
                        self.send_response(200)
                        self.send_header('Content-Type', 'text/event-stream')
                        self.send_header('Cache-Control', 'no-cache')
                        self.send_header('Connection', 'keep-alive')
                        self.end_headers()
                        try:
                            partial_len = 0
                            for partial in sess.respond(user_msg):
                                delta = partial[partial_len:]
                                partial_len = len(partial)
                                if not delta: continue
                                chunk = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": req.get("model", "nexa-model"),
                                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]
                                }
                                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
                                self.wfile.flush()
                            chunk = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": req.get("model", "nexa-model"),
                                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode('utf-8'))
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                        except Exception as e:
                            print(f"\n[Stream Error] {e}")
                    else:
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json')
                        self.end_headers()
                        full_resp = ""
                        for partial in sess.respond(user_msg):
                            full_resp = partial
                        res = {
                            "id": completion_id,
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": req.get("model", "nexa-model"),
                            "choices": [{
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": full_resp
                                },
                                "finish_reason": "stop"
                            }]
                        }
                        self.wfile.write(json.dumps(res).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()

        class ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        server_address = ("0.0.0.0", port)
        print(f"\n" + "=" * 50)
        print(f"  Nexa {NEXA_VERSION} API Server (stdlib)")
        print(f"  http://0.0.0.0:{port}/v1")
        print(f"  Install fastapi+uvicorn for full features:")
        print(f"    pip install fastapi uvicorn")
        print("=" * 50 + "\n")
        try:
            httpd = ThreadingHTTPServer(server_address, OpenAIHandler)
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down API server...")
            batch_scheduler.stop()
            session_mgr.close_all()
            httpd.server_close()
def main():
    p = argparse.ArgumentParser(description="Nexa 1.3 API Server")
    p.add_argument("--checkpoint", default="checkpoints/best.pt")
    p.add_argument("--lora-ckpt", default=None)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument(
        "--merge-lora",
        action="store_true",
        help="Merge LoRA weights into base model before inference",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--min-p", type=float, default=0.05)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--cli", action="store_true")
    p.add_argument("--finetune", action="store_true")
    p.add_argument(
        "--data",
        type=str,
        default="philschmid/dolly-15k-oai-style",
        help="HF dataset or local JSONL",
    )
    p.add_argument("--sft-epochs", type=int, default=3)
    p.add_argument("--sft-lr", type=float, default=2e-4)
    p.add_argument("--fast-mode", action="store_true")
    p.add_argument("--build-self-improve", action="store_true")
    p.add_argument("--fail-log", default=FAIL_LOG_PATH)
    p.add_argument("--self-improve-out", default=SELF_IMPROVE_DATASET_PATH)
    p.add_argument("--self-improve-samples", type=int, default=200)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()
    assert os.path.exists(args.checkpoint), f"No checkpoint: {args.checkpoint}."
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", Config(vocab_size=vocab_size))
    if hasattr(config, "vocab_size") and config.vocab_size != vocab_size:
        raise AssertionError(
            f"Checkpoint vocab_size={config.vocab_size} != tokenizer vocab_size={vocab_size}"
        )
    config.vocab_size = vocab_size
    if not hasattr(config, "eos_id") or config.eos_id is None:
        config.eos_id = tokenizer.token_to_id(EOS_TOKEN)
    print(f"Loading model from {args.checkpoint}...")
    model = NexaModel(config).to(device)
    safe_load_model_state(model, ckpt["model"], label="chat checkpoint")
    print(f"Loaded (val_loss={ckpt.get('val_loss', '?')})")
    if args.finetune:
        sft_finetune(
            model,
            tokenizer,
            args.data,
            config,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            epochs=args.sft_epochs,
            lr=args.sft_lr,
        )
        return
    if args.build_self_improve:
        build_self_improve_dataset(
            model,
            tokenizer,
            device,
            fail_log_path=args.fail_log,
            out_path=args.self_improve_out,
            max_cases=args.self_improve_samples,
            critic_model=clone_critic_model(model, device),
        )
        return
    if args.lora_ckpt:
        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        load_lora(model, args.lora_ckpt)
        if args.merge_lora:
            merge_lora(model)
    model.eval()
    critic_model = clone_critic_model(model, device)
    if args.cli:
        print("\n" + "=" * 50)
        print(f"  Nexa {NEXA_VERSION} (CLI)")
        print("  /reset to clear, /quit to exit")
        print("=" * 50)
        session = ChatSession(
            model,
            tokenizer,
            device,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_tokens,
            fast_mode=args.fast_mode,
            critic_model=critic_model,
        )
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if user_input.lower() in ("/quit", "quit", "exit", "q"):
                print("Bye!")
                break
            if user_input.lower() == "/reset":
                session.reset()
                print("Chat cleared.")
                continue
            if not user_input:
                continue
            print("\nNexa: ", end="", flush=True)
            response = ""
            for partial in session.respond(user_input):
                response = partial
                print(f"\rNexa: {response}", end="", flush=True)
            print()
    else:
        launch_openai_api(
            model,
            tokenizer,
            device,
            critic_model=critic_model,
            port=args.port,
        )
if __name__ == "__main__":
    main()
