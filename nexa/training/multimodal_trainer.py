"""Multimodal trainer for Nexa 1.5."""
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from nexa.training.contrastive_loss import ContrastiveLoss
from nexa.training.collapse_detector import CollapseEarlyStopping
from nexa.training.image_usage_tracker import ImageUsageTracker


class MultimodalTrainer:
    def __init__(self, model, tokenizer, config, collapse_detector, curriculum_loader):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.collapse_detector = collapse_detector
        self.curriculum_loader = curriculum_loader
        self.current_epoch = 0
        self.global_step = 0

        # Loss components
        self.contrastive_loss = ContrastiveLoss()
        self.max_grad_norm = 1.0

        # Adaptive collapse check interval
        self.collapse_check_interval = self._get_check_interval()

        # Early stopping
        self.early_stopping = CollapseEarlyStopping(patience=3, threshold=0.85)

        # Logging
        log_dir = getattr(config, 'log_dir', 'logs/multimodal')
        self.writer = SummaryWriter(log_dir)
        self.image_usage_tracker = ImageUsageTracker()

        # Optimizer and schedulers
        total_steps = getattr(config, 'max_iters', 10000)
        self.optimizers = self._configure_optimizer_with_scheduler(total_steps)

    def _get_check_interval(self):
        batch_size = self.config.batch_size
        if batch_size < 8:
            return 200
        elif batch_size <= 32:
            return 300
        else:
            return 500

    def _configure_optimizer_with_scheduler(self, total_steps):
        from nexa.vision.projector import VisionProjector
        from nexa.vision.image_text_gate import ImageTextGate
        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

        projector_params = []
        lora_params = []
        seen = set()

        # Type-based param grouping
        for name, module in self.model.named_modules():
            if isinstance(module, (VisionProjector, ImageTextGate)):
                for param in module.parameters():
                    if param.requires_grad and id(param) not in seen:
                        projector_params.append(param)
                        seen.add(id(param))
            elif hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                for param in module.parameters():
                    if param.requires_grad and id(param) not in seen:
                        lora_params.append(param)
                        seen.add(id(param))

        optimizer_projector = torch.optim.AdamW(projector_params, lr=5e-4, betas=(0.9, 0.95))

        if lora_params:
            optimizer_lora = torch.optim.AdamW(lora_params, lr=2e-4, betas=(0.9, 0.95))
        else:
            optimizer_lora = None

        # Projector scheduler: warmup 500 steps
        warmup_projector = LinearLR(optimizer_projector, start_factor=0.1, total_iters=min(500, total_steps))
        cosine_projector = CosineAnnealingLR(optimizer_projector, T_max=max(1, total_steps - 500))
        scheduler_projector = SequentialLR(
            optimizer_projector,
            schedulers=[warmup_projector, cosine_projector],
            milestones=[min(500, total_steps)]
        )

        if optimizer_lora is not None:
            # LoRA scheduler: warmup 1000 steps
            warmup_lora = LinearLR(optimizer_lora, start_factor=0.1, total_iters=min(1000, total_steps))
            cosine_lora = CosineAnnealingLR(optimizer_lora, T_max=max(1, total_steps - 1000))
            scheduler_lora = SequentialLR(
                optimizer_lora,
                schedulers=[warmup_lora, cosine_lora],
                milestones=[min(1000, total_steps)]
            )
        else:
            scheduler_lora = None

        return {
            'optimizer_projector': optimizer_projector,
            'optimizer_lora': optimizer_lora,
            'scheduler_projector': scheduler_projector,
            'scheduler_lora': scheduler_lora,
        }

    def get_contrastive_weight(self, epoch):
        return 0.05 if epoch < 2 else 0.1

    def get_gate_target(self, epoch):
        return 0.7 if epoch < 2 else 0.8

    def compute_loss(self, outputs, targets):
        ce_loss = outputs['loss'] if outputs['loss'] is not None else 0.0
        total_loss = ce_loss
        loss_dict = {'ce_loss': ce_loss.item() if torch.is_tensor(ce_loss) else ce_loss}

        # Contrastive loss
        if outputs['image_features'] is not None and outputs['text_features'] is not None:
            image_pooled = outputs['image_features'].mean(dim=1)
            text_pooled = outputs['text_features'].mean(dim=1)
            contrastive = self.contrastive_loss(image_pooled, text_pooled)
            weight = self.get_contrastive_weight(self.current_epoch)
            total_loss = total_loss + weight * contrastive
            loss_dict['contrastive'] = contrastive.item()
            loss_dict['contrastive_weight'] = weight

        # Gate regularization
        if outputs['gate_alpha'] is not None:
            target = self.get_gate_target(self.current_epoch)
            gate_reg = 0.01 * (outputs['gate_alpha'] - target).pow(2).mean()
            total_loss = total_loss + gate_reg
            loss_dict['gate_reg'] = gate_reg.item()
            loss_dict['gate_alpha'] = outputs['gate_alpha'].mean().item()
            loss_dict['gate_target'] = target

        return total_loss, loss_dict

    def train_step(self, batch):
        self.model.train()

        outputs = self.model(
            input_ids=batch['input_ids'],
            labels=batch['labels'],
            images=batch.get('images'),
        )

        loss, loss_dict = self.compute_loss(outputs, batch['labels'])
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # Optimizer steps
        self.optimizers['optimizer_projector'].step()
        self.optimizers['optimizer_projector'].zero_grad()
        self.optimizers['scheduler_projector'].step()

        if self.optimizers['optimizer_lora'] is not None:
            self.optimizers['optimizer_lora'].step()
            self.optimizers['optimizer_lora'].zero_grad()
            self.optimizers['scheduler_lora'].step()

        # DDP-safe gate step
        model_unwrapped = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model_unwrapped, 'gate'):
            model_unwrapped.gate.step()

        # Logging
        self.writer.add_scalar('train/grad_norm', grad_norm.item(), self.global_step)
        self.writer.add_scalar('train/loss', loss.item(), self.global_step)
        self.writer.add_scalar('train/lr_projector', self.optimizers['optimizer_projector'].param_groups[0]['lr'], self.global_step)

        if self.optimizers['optimizer_lora'] is not None:
            self.writer.add_scalar('train/lr_lora', self.optimizers['optimizer_lora'].param_groups[0]['lr'], self.global_step)

        for key, value in loss_dict.items():
            self.writer.add_scalar(f'train/{key}', value, self.global_step)

        self.global_step += 1
        loss_dict['grad_norm'] = grad_norm.item()
        return loss_dict

    def train_epoch(self, epoch):
        self.current_epoch = epoch
        data = self.curriculum_loader.get_epoch_data(epoch)

        for step, batch in enumerate(data):
            loss_dict = self.train_step(batch)

            if step % 100 == 0:
                print(f"Epoch {epoch} Step {step}: {loss_dict}")

            # Collapse detection and monitoring
            if step % self.collapse_check_interval == 0 and batch.get('images') is not None:
                similarity = self.collapse_detector.test_image_usage(
                    batch['images'][0],
                    batch.get('prompt', [''])[0]
                )
                self.writer.add_scalar('eval/collapse_similarity', similarity, self.global_step)

                # Image usage score
                usage_score = self.image_usage_tracker.compute_image_usage_score(
                    self.model,
                    batch['images'][0:1],
                    batch['input_ids'][0:1]
                )
                self.writer.add_scalar('eval/image_usage_score', usage_score, self.global_step)

                if self.early_stopping.check(similarity):
                    print("Training stopped due to persistent collapse")
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
