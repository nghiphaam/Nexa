"""Multimodal trainer for Nexa 1.6."""
from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter

from nexa.training.contrastive_loss import ContrastiveLoss
from nexa.training.collapse_detector import CollapseEarlyStopping
from nexa.training.image_usage_tracker import ImageUsageTracker


class MultimodalTrainer:
    """Train multimodal Nexa models with AMP, curriculum scheduling, and collapse detection."""

    def __init__(self, model, tokenizer, config, collapse_detector, curriculum_loader):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.collapse_detector = collapse_detector
        self.curriculum_loader = curriculum_loader
        self.current_epoch = 0
        self.global_step = 0
        self.contrastive_loss = ContrastiveLoss()
        self.max_grad_norm = getattr(config, "grad_clip", 1.0)
        self.collapse_check_interval = self._get_check_interval()
        self.early_stopping = CollapseEarlyStopping(
            patience=getattr(config, "collapse_patience", 3),
            threshold=getattr(config, "collapse_threshold", 0.85),
        )
        log_dir = getattr(config, "log_dir", "logs/nexa-1.6/multimodal")
        self.writer = SummaryWriter(log_dir)
        self.image_usage_tracker = ImageUsageTracker()
        total_steps = getattr(config, "max_iters", 10000)
        self.optimizers = self._configure_optimizer_with_scheduler(total_steps)
        self.scaler = self._create_grad_scaler()

    def _model_device(self):
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(getattr(self.config, "device", "cpu"))

    def _create_grad_scaler(self):
        if self._model_device().type != "cuda":
            return None
        use_fp16 = getattr(self.config, "use_fp16", False) or getattr(self.config, "dtype", "") == "float16"
        return torch.cuda.amp.GradScaler(enabled=use_fp16)

    def _autocast_context(self):
        if self._model_device().type != "cuda":
            return contextlib.nullcontext()
        dtype_name = getattr(self.config, "dtype", "float32")
        if dtype_name == "float16":
            dtype = torch.float16
        elif dtype_name == "bfloat16":
            dtype = torch.bfloat16
        else:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=dtype)

    def _get_check_interval(self):
        batch_size = self.config.batch_size
        if batch_size < 8:
            return 200
        if batch_size <= 32:
            return 300
        return 500

    def _configure_optimizer_with_scheduler(self, total_steps):
        from nexa.vision.projector import VisionProjector
        from nexa.vision.image_text_gate import ImageTextGate
        from nexa.vision.cross_modal_attention import CrossModalAttentionFusion
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        projector_params = []
        fusion_params = []
        lora_params = []
        fallback_params = []
        seen = set()

        for _, module in self.model.named_modules():
            if isinstance(module, (VisionProjector, ImageTextGate)):
                for param in module.parameters():
                    if param.requires_grad and id(param) not in seen:
                        projector_params.append(param)
                        seen.add(id(param))
            elif isinstance(module, CrossModalAttentionFusion):
                for param in module.parameters():
                    if param.requires_grad and id(param) not in seen:
                        fusion_params.append(param)
                        seen.add(id(param))
            elif hasattr(module, "lora_A") or hasattr(module, "lora_B"):
                for param in module.parameters():
                    if param.requires_grad and id(param) not in seen:
                        lora_params.append(param)
                        seen.add(id(param))

        for param in self.model.parameters():
            if param.requires_grad and id(param) not in seen:
                fallback_params.append(param)
                seen.add(id(param))

        if not projector_params and not fusion_params and fallback_params:
            projector_params = fallback_params
            fallback_params = []

        projector_group = projector_params + fusion_params
        if not projector_group:
            raise ValueError("No trainable projector/fusion parameters found for multimodal training")
        optimizer_projector = torch.optim.AdamW(projector_group, lr=5e-4, betas=(0.9, 0.95))

        extra_trainable = lora_params + fallback_params
        optimizer_lora = None
        if extra_trainable:
            optimizer_lora = torch.optim.AdamW(extra_trainable, lr=2e-4, betas=(0.9, 0.95))

        warmup_projector = LinearLR(optimizer_projector, start_factor=0.1, total_iters=min(500, total_steps))
        cosine_projector = CosineAnnealingLR(optimizer_projector, T_max=max(1, total_steps - 500))
        scheduler_projector = SequentialLR(
            optimizer_projector,
            schedulers=[warmup_projector, cosine_projector],
            milestones=[min(500, total_steps)],
        )

        scheduler_lora = None
        if optimizer_lora is not None:
            warmup_lora = LinearLR(optimizer_lora, start_factor=0.1, total_iters=min(1000, total_steps))
            cosine_lora = CosineAnnealingLR(optimizer_lora, T_max=max(1, total_steps - 1000))
            scheduler_lora = SequentialLR(
                optimizer_lora,
                schedulers=[warmup_lora, cosine_lora],
                milestones=[min(1000, total_steps)],
            )

        return {
            "optimizer_projector": optimizer_projector,
            "optimizer_lora": optimizer_lora,
            "scheduler_projector": scheduler_projector,
            "scheduler_lora": scheduler_lora,
        }

    def get_contrastive_weight(self, epoch):
        return 0.05 if epoch < 2 else 0.1

    def get_gate_target(self, epoch):
        return 0.7 if epoch < 2 else 0.8

    def compute_loss(self, outputs, targets):
        ce_loss = outputs["loss"] if outputs["loss"] is not None else 0.0
        total_loss = ce_loss
        loss_dict = {"ce_loss": ce_loss.item() if torch.is_tensor(ce_loss) else float(ce_loss)}

        if outputs.get("image_features") is not None and outputs.get("text_features") is not None:
            image_pooled = outputs["image_features"].mean(dim=1)
            text_pooled = outputs["text_features"].mean(dim=1)
            contrastive = self.contrastive_loss(image_pooled, text_pooled)
            weight = self.get_contrastive_weight(self.current_epoch)
            total_loss = total_loss + weight * contrastive
            loss_dict["contrastive"] = contrastive.item()
            loss_dict["contrastive_weight"] = weight

        if outputs.get("gate_alpha") is not None:
            target = self.get_gate_target(self.current_epoch)
            gate_reg = 0.01 * (outputs["gate_alpha"] - target).pow(2).mean()
            total_loss = total_loss + gate_reg
            loss_dict["gate_reg"] = gate_reg.item()
            loss_dict["gate_alpha"] = outputs["gate_alpha"].mean().item()
            loss_dict["gate_target"] = target

        return total_loss, loss_dict

    def _get_eos_id(self) -> int:
        eos_id = getattr(self.config, "eos_id", None)
        if eos_id is not None:
            return int(eos_id)
        token_to_id = getattr(self.tokenizer, "token_to_id", None)
        if callable(token_to_id):
            for token in ("<|endoftext|>", "</s>"):
                try:
                    value = token_to_id(token)
                except (KeyError, TypeError, ValueError):
                    continue
                if value is not None:
                    return int(value)
        return 0

    def _extract_image(self, sample: dict[str, Any]):
        if "images" in sample:
            return sample["images"]
        if "image" in sample:
            return sample["image"]
        return None

    def _build_text_from_sample(self, sample: dict[str, Any]) -> str:
        if "text" in sample and sample["text"]:
            return str(sample["text"])

        if "messages" in sample and sample["messages"]:
            parts = []
            for msg in sample["messages"]:
                role = msg.get("role", "user")
                content = str(msg.get("content", ""))
                if role == "user":
                    parts.append(f"<|user|>{content}<|assistant|>")
                elif role == "assistant":
                    parts.append(f"{content}<|endoftext|>")
                else:
                    parts.append(content)
            text = "".join(parts).strip()
            if text:
                return text

        prompt = str(sample.get("prompt") or sample.get("question") or sample.get("instruction") or "")
        response = str(sample.get("response") or sample.get("answer") or sample.get("caption") or "")
        if prompt and response:
            return f"<|user|>{prompt}<|assistant|>{response}<|endoftext|>"
        if prompt:
            return f"<|user|>{prompt}<|assistant|>"
        if response:
            return f"<|assistant|>{response}<|endoftext|>"
        raise ValueError("Could not build multimodal training text from sample")

    def _collate_images(self, images):
        present = [img for img in images if img is not None]
        if not present:
            return None
        if len(present) != len(images):
            raise ValueError("Mixed multimodal batch with missing images is not supported")
        if all(torch.is_tensor(img) for img in present):
            tensors = [img if img.dim() == 4 else img.unsqueeze(0) for img in present]
            return torch.cat(tensors, dim=0).to(self._model_device())
        return present if len(present) > 1 else present[0]

    def _build_autoregressive_batch(self, token_rows, pad_id: int, device):
        if not token_rows:
            raise ValueError("Empty token batch")

        max_len = max(len(ids) for ids in token_rows)
        if max_len <= 0:
            raise ValueError("Token rows must not be empty")

        padded_ids = []
        padded_labels = []
        for ids in token_rows:
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [pad_id] * pad_len)
            shifted_labels = ids[1:] + [-100]
            padded_labels.append(shifted_labels + [-100] * pad_len)

        input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)
        return input_ids, labels

    def _prepare_batch(self, batch):
        device = self._model_device()

        if isinstance(batch, dict) and "input_ids" in batch and "labels" in batch:
            prepared = dict(batch)
            prepared["input_ids"] = batch["input_ids"].to(device)
            prepared["labels"] = batch["labels"].to(device)
            if prepared.get("images") is None and prepared.get("image") is not None:
                prepared["images"] = prepared.pop("image")
            if torch.is_tensor(prepared.get("images")):
                prepared["images"] = prepared["images"].to(device)
            return prepared

        samples = batch if isinstance(batch, list) else [batch]
        texts = [self._build_text_from_sample(sample) for sample in samples]
        token_rows = [self.tokenizer.encode(text).ids[: getattr(self.config, "block_size", 512)] for text in texts]
        if not token_rows:
            raise ValueError("Empty multimodal batch")

        eos_id = self._get_eos_id()
        input_ids, labels = self._build_autoregressive_batch(token_rows, eos_id, device)
        prompts = [
            str(sample.get("prompt") or sample.get("question") or sample.get("instruction") or "")
            for sample in samples
        ]
        images = self._collate_images([self._extract_image(sample) for sample in samples])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": images,
            "prompt": prompts,
            "raw_samples": samples,
        }

    def _first_image_for_eval(self, images):
        if images is None:
            return None
        if torch.is_tensor(images):
            return images[0:1]
        if isinstance(images, (list, tuple)):
            return images[0]
        return images

    def train_step(self, batch):
        batch = self._prepare_batch(batch)
        if "input_ids" not in batch or "labels" not in batch:
            raise ValueError("Batch must include 'input_ids' and 'labels'")

        self.model.train()
        self.optimizers["optimizer_projector"].zero_grad(set_to_none=True)
        if self.optimizers["optimizer_lora"] is not None:
            self.optimizers["optimizer_lora"].zero_grad(set_to_none=True)

        with self._autocast_context():
            outputs = self.model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                images=batch.get("images"),
            )
            loss, loss_dict = self.compute_loss(outputs, batch["labels"])

        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizers["optimizer_projector"])
            if self.optimizers["optimizer_lora"] is not None:
                self.scaler.unscale_(self.optimizers["optimizer_lora"])
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizers["optimizer_projector"])
            if self.optimizers["optimizer_lora"] is not None:
                self.scaler.step(self.optimizers["optimizer_lora"])
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizers["optimizer_projector"].step()
            if self.optimizers["optimizer_lora"] is not None:
                self.optimizers["optimizer_lora"].step()

        self.optimizers["scheduler_projector"].step()
        if self.optimizers["scheduler_lora"] is not None:
            self.optimizers["scheduler_lora"].step()

        model_unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_unwrapped, "gate"):
            model_unwrapped.gate.step()

        self.writer.add_scalar("train/grad_norm", grad_norm.item(), self.global_step)
        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
        self.writer.add_scalar(
            "train/lr_projector",
            self.optimizers["optimizer_projector"].param_groups[0]["lr"],
            self.global_step,
        )
        if self.optimizers["optimizer_lora"] is not None:
            self.writer.add_scalar(
                "train/lr_lora",
                self.optimizers["optimizer_lora"].param_groups[0]["lr"],
                self.global_step,
            )
        for key, value in loss_dict.items():
            self.writer.add_scalar(f"train/{key}", value, self.global_step)

        self.global_step += 1
        loss_dict["grad_norm"] = grad_norm.item()
        loss_dict["prepared_batch"] = batch
        return loss_dict

    def train_epoch(self, epoch):
        self.current_epoch = epoch
        data = self.curriculum_loader.get_epoch_data(epoch)
        log_every = max(1, getattr(self.config, "log_interval", 50))
        batch_size = max(1, int(getattr(self.config, "batch_size", 1)))

        for step, batch_start in enumerate(range(0, len(data), batch_size)):
            raw_batch = data[batch_start: batch_start + batch_size]
            loss_dict = self.train_step(raw_batch)
            batch = loss_dict["prepared_batch"]
            if step % log_every == 0:
                printable = {k: v for k, v in loss_dict.items() if k != "prepared_batch"}
                print(f"[epoch {epoch + 1}] step {step}: {printable}")

            images = batch.get("images")
            if step % self.collapse_check_interval == 0 and images is not None:
                image_for_eval = self._first_image_for_eval(images)
                prompt_for_eval = batch.get("prompt", [""])[0] if batch.get("prompt") else ""
                similarity = self.collapse_detector.test_image_usage(image_for_eval, prompt_for_eval)
                self.writer.add_scalar("eval/collapse_similarity", similarity, self.global_step)
                usage_score = self.image_usage_tracker.compute_image_usage_score(
                    self.model,
                    image_for_eval,
                    batch["input_ids"][0:1],
                )
                self.writer.add_scalar("eval/image_usage_score", usage_score, self.global_step)
                if getattr(self.config, "enable_early_stopping", True) and self.early_stopping.check(similarity):
                    print("Training stopped due to persistent multimodal collapse")
                    return False
        return True

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            print(f"\n{'=' * 65}")
            print(f"  EPOCH {epoch + 1}/{num_epochs}")
            print(f"{'=' * 65}")
            should_continue = self.train_epoch(epoch)
            if not should_continue:
                break
