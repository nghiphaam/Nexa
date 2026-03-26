import json
import math
import os
import random
import importlib
import time
from itertools import islice

import numpy as np


HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATASET_MANIFEST = "dataset_manifest.json"
DATASET_SNAPSHOT_DIRNAME = "dataset_snapshot"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"
SYS_TOKEN = "<|system|>"
USR_TOKEN = "<|user|>"
AST_TOKEN = "<|assistant|>"


def require_pkg(name, pip_name=None):
    try:
        importlib.import_module(name)
    except ImportError as exc:
        package = pip_name or name
        raise RuntimeError(
            f"Missing required dependency '{package}'. Install it before running pre_train.py."
        ) from exc


require_pkg("datasets")
require_pkg("huggingface_hub")
require_pkg("tiktoken")


class NexaTokenizer:
    def __init__(self):
        import tiktoken
        import re

        self.enc = tiktoken.get_encoding("gpt2")  # Using GPT-2 BPE compatible encoding
        self._base_vocab_size = self.enc.n_vocab  # 50257
        self._vocab_size = 50261
        self._eos_id = self.enc.eot_token  # 50256

        self.special_map = {
            EOS_TOKEN: 50256,
            PAD_TOKEN: 50257,
            SYS_TOKEN: 50258,
            USR_TOKEN: 50259,
            AST_TOKEN: 50260,
        }
        self.id_to_special = {v: k for k, v in self.special_map.items()}
        pattern = "|".join(re.escape(k) for k in self.special_map.keys())
        self.special_pattern = re.compile(f"({pattern})")

    def get_vocab_size(self):
        return self._vocab_size

    def token_to_id(self, token):
        if token in self.special_map:
            return self.special_map[token]
        raise ValueError(f"Unknown token: {token}")

    def _normalize_text(self, text):
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        return text.replace("\r\n", "\n")

    def encode(self, text):
        normalized = self._normalize_text(text)
        if not normalized:
            return type("E", (), {"ids": []})()
        parts = self.special_pattern.split(normalized)
        ids = []
        for p in parts:
            if p in self.special_map:
                ids.append(self.special_map[p])
            elif p:
                ids.extend(self.enc.encode_ordinary(p))
        return type("E", (), {"ids": ids})()

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]


def _hf_login():
    if not HF_TOKEN:
        return
    try:
        from huggingface_hub import login

        login(token=HF_TOKEN, add_to_git_credential=False)
    except Exception as e:
        print(f"[warn] HF login failed: {e}")


def load_training_dataset(
    dataset_repo: str,
    dataset_name: str | None,
    split: str,
    num_proc: int,
    streaming: bool = False,
):
    from datasets import load_dataset

    if not dataset_repo:
        raise ValueError(
            "dataset_repo is required. Nexa does not set a default dataset."
        )
    label = dataset_repo if not dataset_name else f"{dataset_repo} ({dataset_name})"
    print(f"Loading dataset: {label}")
    kwargs = {"split": split, "streaming": streaming}
    if not streaming:
        kwargs["num_proc"] = num_proc
    if dataset_name is not None:
        kwargs["name"] = dataset_name
    ds = load_dataset(dataset_repo, **kwargs)
    return ds, label, dataset_repo, dataset_name


def iter_stream_text_batches(ds, max_samples: int, batch_size: int = 10000):
    batch = []
    stream = ds if max_samples <= 0 else islice(ds, max_samples)
    for item in stream:
        text = item.get("text", "")
        if not text:
            continue
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def estimate_avg_tokens(ds, tokenizer, n: int = 100) -> float:
    total_tokens = 0
    count = 0
    stream = ds if n <= 0 else islice(ds, n)
    for item in stream:
        text = item.get("text", "")
        if not text:
            continue
        total_tokens += len(tokenizer.encode(text).ids)
        count += 1
    return (total_tokens / count) if count > 0 else 200.0


def _flush_token_buffer(handle, buffer, np_dtype):
    if not buffer:
        return 0
    arr = np.asarray(buffer, dtype=np_dtype)
    handle.write(arr.tobytes())
    n = len(buffer)
    buffer.clear()
    return n


def stream_tokenize_to_bins(
    ds,
    tokenizer,
    eos_id,
    train_path,
    val_path,
    np_dtype,
    max_samples: int = 0,
    flush_tokens: int = 5_000_000,
    val_ratio: float = 0.01,
):
    t0 = time.time()
    train_buffer = []
    val_buffer = []
    docs_seen = 0
    train_docs = 0
    val_docs = 0
    train_tokens = 0
    val_tokens = 0
    rng = random.Random(42)

    def tokenize_batch(texts):
        results = []
        for text in texts:
            if text and text.strip():
                ids = tokenizer.encode(text).ids
                if ids:
                    ids.append(eos_id)
                    results.append(ids)
        return results

    with open(train_path, "wb") as train_f, open(val_path, "wb") as val_f:
        for batch in iter_stream_text_batches(ds, max_samples, batch_size=10000):
            batch_ids = tokenize_batch(batch)

            for ids in batch_ids:
                docs_seen += 1
                if rng.random() < val_ratio:
                    val_buffer.extend(ids)
                    val_docs += 1
                else:
                    train_buffer.extend(ids)
                    train_docs += 1

            if len(train_buffer) >= flush_tokens:
                train_tokens += _flush_token_buffer(train_f, train_buffer, np_dtype)
            if len(val_buffer) >= max(500_000, flush_tokens // 10):
                val_tokens += _flush_token_buffer(val_f, val_buffer, np_dtype)

            if docs_seen % 10000 == 0:
                elapsed = time.time() - t0
                rate = docs_seen / elapsed if elapsed > 0 else 0
                print(
                    f"  docs={docs_seen:,} ({rate:.0f}/s) train={train_docs:,} val={val_docs:,} "
                    f"train_tok={train_tokens + len(train_buffer):,} val_tok={val_tokens + len(val_buffer):,} "
                    f"| {elapsed:.0f}s",
                    end="\r",
                )

        train_tokens += _flush_token_buffer(train_f, train_buffer, np_dtype)
        val_tokens += _flush_token_buffer(val_f, val_buffer, np_dtype)

    print()
    return {
        "docs_seen": docs_seen,
        "train_docs": train_docs,
        "val_docs": val_docs,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "elapsed_s": time.time() - t0,
    }


def get_system_ram_gb():
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        return (page_size * phys_pages) / (1024**3)
    except Exception:
        try:
            import psutil

            return psutil.virtual_memory().total / (1024**3)
        except Exception:
            return 0.0


def estimate_dataset_ram_gb(
    max_samples: int, avg_tokens: float = 200.0, bytes_per_token: int = 2
) -> float:
    if max_samples <= 0:
        return float("inf")
    return (max_samples * avg_tokens * bytes_per_token) / (1024**3)


def should_stream_data_prep(
    max_samples: int, ram_gb: float, avg_tokens: float = 200.0
) -> bool:
    if ram_gb <= 0:
        return True
    est_gb = estimate_dataset_ram_gb(max_samples, avg_tokens=avg_tokens)
    if math.isfinite(est_gb):
        return est_gb >= (ram_gb * 0.6)
    return ram_gb < 96


def compute_val_ratio(max_samples: int) -> float:
    if max_samples > 0:
        return min(0.01, max(1000 / max_samples, 0.001))
    return 0.001


def tokenize_dataset_in_memory(
    ds_train, tokenizer, eos_id, data_dir, np_dtype, ram_gb: float, val_ratio: float
):
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    total = len(ds_train)
    val_size = min(total - 1, max(1, int(total * val_ratio)))
    ds_split = ds_train.train_test_split(test_size=val_size, seed=42)
    print(
        f"Train: {len(ds_split['train']):,} docs | Val: {len(ds_split['test']):,} docs"
    )

    def fast_tokenize(ds, out_path):
        t0 = time.time()
        chunk_size = max(5000, min(500000, int(max(1.0, ram_gb) * 50000)))

        def tokenize_chunk(texts):
            encodings = tokenizer.encode_batch(texts)
            total_len = sum(len(enc.ids) + 1 for enc in encodings if enc is not None)
            if total_len == 0:
                return None
            chunk = np.empty(total_len, dtype=np_dtype)
            pos = 0
            for enc in encodings:
                if enc is None:
                    continue
                tok_len = len(enc.ids)
                chunk[pos : pos + tok_len] = enc.ids
                chunk[pos + tok_len] = eos_id
                pos += tok_len + 1
            return chunk

        with open(out_path, "wb") as f:
            for i in range(0, len(ds), chunk_size):
                texts = ds[i : i + chunk_size]["text"]
                chunk = tokenize_chunk(texts)
                if chunk is not None:
                    f.write(chunk.tobytes())

                elapsed = time.time() - t0
                rate = (i + len(texts)) / elapsed if elapsed > 0 else 0
                print(
                    f"  {min(i + chunk_size, len(ds)):,}/{len(ds):,} docs ({rate:.0f}/s) | chunk={chunk_size:,} | {elapsed:.0f}s",
                    end="\r",
                )
        mb = os.path.getsize(out_path) / 1e6
        print(f"\n  Saved {out_path} ({mb:.0f}MB, {time.time() - t0:.1f}s)")

    print("\nIn-memory tokenizing train...")
    fast_tokenize(ds_split["train"], train_path)
    print("In-memory tokenizing val...")
    fast_tokenize(ds_split["test"], val_path)


def resolve_data_plan(
    dataset_repo: str,
    dataset_name: str | None = None,
    tokenizer=None,
    max_samples: int = 0,
    force_stream: bool = False,
    force_in_memory: bool = False,
):
    ram_gb = get_system_ram_gb()
    avg_tokens = 200.0
    n_proc = max(1, (os.cpu_count() or 2) - 1)
    if tokenizer is not None:
        probe_split = (
            "train" if max_samples <= 0 else f"train[:{max(max_samples, 100)}]"
        )
        try:
            ds_probe, _label, _repo, _subset = load_training_dataset(
                dataset_repo,
                dataset_name,
                split=probe_split,
                num_proc=n_proc,
                streaming=True,
            )
            avg_tokens = estimate_avg_tokens(ds_probe, tokenizer, n=100)
        except Exception:
            avg_tokens = 200.0
    est_ram_gb = estimate_dataset_ram_gb(max_samples, avg_tokens=avg_tokens)
    use_streaming = should_stream_data_prep(
        max_samples=max_samples, ram_gb=ram_gb, avg_tokens=avg_tokens
    )
    if force_stream:
        use_streaming = True
    elif force_in_memory:
        use_streaming = False
    val_ratio = compute_val_ratio(max_samples)
    return {
        "ram_gb": ram_gb,
        "avg_tokens": avg_tokens,
        "est_ram_gb": est_ram_gb,
        "use_streaming": use_streaming,
        "val_ratio": val_ratio,
        "num_proc": n_proc,
    }


def _write_manifest(data_dir, payload):
    os.makedirs(data_dir, exist_ok=True)
    manifest_path = os.path.join(data_dir, DATASET_MANIFEST)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return manifest_path


def _load_manifest(data_dir):
    manifest_path = os.path.join(data_dir, DATASET_MANIFEST)
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _snapshot_matches_request(data_dir, dataset_repo, dataset_name, split_str, max_samples):
    snapshot_dir = os.path.join(data_dir, DATASET_SNAPSHOT_DIRNAME)
    manifest = _load_manifest(data_dir)
    if manifest is None:
        return False
    return (
        manifest.get("snapshot_dir") == snapshot_dir
        and manifest.get("repo_id") == dataset_repo
        and manifest.get("subset_name") == dataset_name
        and manifest.get("split") == split_str
        and int(manifest.get("max_samples", 0)) == int(max_samples)
        and os.path.exists(snapshot_dir)
    )


def download_dataset(
    data_dir,
    dataset_repo: str,
    dataset_name: str | None = None,
    max_samples: int = 0,
    force_stream: bool = False,
    force_in_memory: bool = False,
    tokenizer=None,
):
    os.makedirs(data_dir, exist_ok=True)
    _hf_login()
    plan = resolve_data_plan(
        dataset_repo,
        dataset_name,
        tokenizer=tokenizer,
        max_samples=max_samples,
        force_stream=force_stream,
        force_in_memory=force_in_memory,
    )
    split_str = "train" if max_samples <= 0 else f"train[:{max_samples}]"
    ds, label, repo_id, subset_name = load_training_dataset(
        dataset_repo,
        dataset_name,
        split=split_str,
        num_proc=plan["num_proc"],
        streaming=plan["use_streaming"],
    )

    manifest = {
        "brand": "Nexa",
        "source_label": label,
        "repo_id": repo_id,
        "subset_name": subset_name,
        "split": split_str,
        "streaming": plan["use_streaming"],
        "max_samples": max_samples,
        "ram_gb": plan["ram_gb"],
        "avg_tokens": plan["avg_tokens"],
        "est_ram_gb": None
        if not math.isfinite(plan["est_ram_gb"])
        else plan["est_ram_gb"],
        "val_ratio": plan["val_ratio"],
        "downloaded_at": time.time(),
    }

    snapshot_dir = os.path.join(data_dir, DATASET_SNAPSHOT_DIRNAME)
    if not plan["use_streaming"]:
        try:
            manifest["rows"] = len(ds)
        except Exception:
            manifest["rows"] = None
        try:
            _ = ds[0]
        except Exception:
            pass
        if max_samples > 0 and not os.path.exists(snapshot_dir):
            ds.save_to_disk(snapshot_dir)
            manifest["snapshot_dir"] = snapshot_dir
    else:
        manifest["probed_docs"] = sum(1 for _ in islice(ds, 256))

    manifest_path = _write_manifest(data_dir, manifest)
    print(
        f"[dataset] source={label} mode={'stream' if plan['use_streaming'] else 'in-memory'}"
    )
    print(f"[dataset] manifest={manifest_path}")
    if manifest.get("snapshot_dir"):
        print(f"[dataset] snapshot={manifest['snapshot_dir']}")
    return manifest


def prepare_data(
    data_dir,
    tokenizer,
    eos_id,
    vocab_size,
    dataset_repo: str,
    dataset_name: str | None = None,
    max_samples: int = 0,
    force_stream: bool = False,
    force_in_memory: bool = False,
):
    os.makedirs(data_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    if os.path.exists(train_path) and os.path.exists(val_path):
        t_mb = os.path.getsize(train_path) / 1e6
        v_mb = os.path.getsize(val_path) / 1e6
        print(f"Data already exists: train={t_mb:.0f}MB, val={v_mb:.0f}MB")
        return

    _hf_login()
    plan = resolve_data_plan(
        dataset_repo,
        dataset_name,
        tokenizer=tokenizer,
        max_samples=max_samples,
        force_stream=force_stream,
        force_in_memory=force_in_memory,
    )
    est_str = (
        f"{plan['est_ram_gb']:.1f}GB"
        if math.isfinite(plan["est_ram_gb"])
        else "unknown"
    )
    print(
        f"[data_prep] RAM={plan['ram_gb']:.1f}GB | avg_tokens={plan['avg_tokens']:.1f} | "
        f"est_dataset={est_str} | val_ratio={plan['val_ratio']:.4f} -> mode={'stream' if plan['use_streaming'] else 'in-memory'}"
    )

    split_str = "train" if max_samples <= 0 else f"train[:{max_samples}]"
    snapshot_dir = os.path.join(data_dir, DATASET_SNAPSHOT_DIRNAME)
    if not plan["use_streaming"] and _snapshot_matches_request(
        data_dir,
        dataset_repo,
        dataset_name,
        split_str,
        max_samples,
    ):
        from datasets import load_from_disk

        ds_train = load_from_disk(snapshot_dir)
        dataset_label = "local snapshot"
    else:
        ds_train, dataset_label, repo_id, subset_name = load_training_dataset(
            dataset_repo,
            dataset_name,
            split=split_str,
            num_proc=plan["num_proc"],
            streaming=plan["use_streaming"],
        )
        _write_manifest(
            data_dir,
            {
                "brand": "Nexa",
                "source_label": dataset_label,
                "repo_id": repo_id,
                "subset_name": subset_name,
                "split": split_str,
                "streaming": plan["use_streaming"],
                "max_samples": max_samples,
                "ram_gb": plan["ram_gb"],
                "avg_tokens": plan["avg_tokens"],
                "est_ram_gb": None
                if not math.isfinite(plan["est_ram_gb"])
                else plan["est_ram_gb"],
                "val_ratio": plan["val_ratio"],
                "prepared_at": time.time(),
            },
        )

    print(f"Using dataset: {dataset_label}")
    np_dtype = np.uint16 if vocab_size < 65536 else np.uint32
    if plan["use_streaming"]:
        print("\nStreaming + tokenizing dataset...")
        stats = stream_tokenize_to_bins(
            ds_train,
            tokenizer=tokenizer,
            eos_id=eos_id,
            train_path=train_path,
            val_path=val_path,
            np_dtype=np_dtype,
            max_samples=max_samples,
            val_ratio=plan["val_ratio"],
        )
        train_mb = os.path.getsize(train_path) / 1e6
        val_mb = os.path.getsize(val_path) / 1e6
        print(
            f"Train docs: {stats['train_docs']:,} | Val docs: {stats['val_docs']:,} | "
            f"Train tok: {stats['train_tokens']:,} | Val tok: {stats['val_tokens']:,}"
        )
        print(
            f"Saved train={train_mb:.0f}MB val={val_mb:.0f}MB in {stats['elapsed_s']:.1f}s"
        )
    else:
        tokenize_dataset_in_memory(
            ds_train,
            tokenizer,
            eos_id,
            data_dir,
            np_dtype,
            ram_gb=plan["ram_gb"],
            val_ratio=plan["val_ratio"],
        )

    print("\nData preparation done!")


def main():
    import argparse

    p = argparse.ArgumentParser(description="Nexa pre-train data pipeline")
    p.add_argument("--dataset", required=True, help="Hugging Face dataset repo")
    p.add_argument(
        "--dataset-config", default=None, help="Optional dataset config / subset name"
    )
    p.add_argument("--data-dir", default="data")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--force-stream", action="store_true")
    p.add_argument("--force-in-memory", action="store_true")
    args = p.parse_args()

    tokenizer = NexaTokenizer()
    download_dataset(
        data_dir=args.data_dir,
        dataset_repo=args.dataset,
        dataset_name=args.dataset_config,
        max_samples=args.max_samples,
        force_stream=args.force_stream,
        force_in_memory=args.force_in_memory,
        tokenizer=tokenizer,
    )
    prepare_data(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        eos_id=tokenizer.token_to_id(EOS_TOKEN),
        vocab_size=tokenizer.get_vocab_size(),
        dataset_repo=args.dataset,
        dataset_name=args.dataset_config,
        max_samples=args.max_samples,
        force_stream=args.force_stream,
        force_in_memory=args.force_in_memory,
    )


if __name__ == "__main__":
    main()
