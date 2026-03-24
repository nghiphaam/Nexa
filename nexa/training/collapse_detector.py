"""Collapse detector for multimodal training."""
import torch
import torch.nn.functional as F


class CollapseDetector:
    def __init__(self, model, tokenizer, threshold=0.80, writer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.writer = writer
        self.step = 0

    @torch.no_grad()
    def test_image_usage(self, image, prompt):
        """Test if model actually uses image."""
        # With image
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt).ids], device=next(self.model.parameters()).device)
        out1 = self.model.generate(input_ids=prompt_ids, images=image, max_new_tokens=50)

        # Without image
        out2 = self.model.generate(input_ids=prompt_ids, images=None, max_new_tokens=50)

        # Get embeddings
        emb1 = self._get_sentence_embedding(out1)
        emb2 = self._get_sentence_embedding(out2)

        similarity = F.cosine_similarity(emb1, emb2, dim=0).item()

        if self.writer is not None:
            self.writer.add_scalar('collapse/similarity', similarity, self.step)
            self.writer.add_scalar('collapse/detected', 1.0 if similarity > self.threshold else 0.0, self.step)

        self.step += 1
        return similarity

    def _get_sentence_embedding(self, tokens):
        if isinstance(tokens, torch.Tensor):
            token_tensor = tokens if tokens.dim() == 2 else tokens.unsqueeze(0)
        else:
            token_tensor = torch.tensor([tokens], device=next(self.model.parameters()).device)

        model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model, 'text_model'):
            embeddings = model.text_model.transformer.wte(token_tensor)
        else:
            embeddings = model.transformer.wte(token_tensor)

        return embeddings.mean(dim=1).squeeze(0)


class CollapseEarlyStopping:
    def __init__(self, patience=3, threshold=0.85):
        self.patience = patience
        self.threshold = threshold
        self.collapse_history = []

    def check(self, similarity):
        is_collapse = similarity > self.threshold
        self.collapse_history.append(is_collapse)

        if len(self.collapse_history) > self.patience:
            self.collapse_history.pop(0)

        if len(self.collapse_history) == self.patience and all(self.collapse_history):
            return True

        return False
