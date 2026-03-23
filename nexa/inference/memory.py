"""Vector memory for retrieval-augmented generation."""
import torch


class VectorMemory:
    """NEXA 1.3: Batch cosine similarity using stacked tensor."""
    def __init__(self, max_entries=1024, embed_dim=2048):
        self.max_entries = max_entries
        self.embed_dim = embed_dim
        self.db = []
        self._emb_cache_tensor = None
        self._emb_cache_dirty = True

    def add(self, embedding, kv_data):
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        self.db.append((embedding.detach().cpu(), kv_data))
        if len(self.db) > self.max_entries:
            self.db.pop(0)
        self._emb_cache_dirty = True

    def _rebuild_emb_cache(self, device):
        if not self.db:
            self._emb_cache_tensor = None
            return
        embs = [entry[0].to(device) for entry in self.db]
        self._emb_cache_tensor = torch.cat(embs, dim=0)
        self._emb_cache_dirty = False

    def retrieve_kv(self, query_ids, decay=0.05, top_k=None, min_score=0.2):
        if not self.db:
            return []
        device = query_ids.device if isinstance(query_ids, torch.Tensor) else "cpu"
        if self._emb_cache_dirty:
            self._rebuild_emb_cache(device)
        if self._emb_cache_tensor is None:
            return []
        q = query_ids.float()
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        cache_norm = self._emb_cache_tensor / (self._emb_cache_tensor.norm(dim=-1, keepdim=True) + 1e-8)
        sims = (q_norm @ cache_norm.T).squeeze(0)
        if top_k is not None:
            top_k = min(top_k, len(self.db))
            _, indices = torch.topk(sims, top_k)
            return [(self.db[i][1], sims[i].item()) for i in indices if sims[i].item() >= min_score]
        return [(self.db[i][1], sims[i].item()) for i in range(len(self.db)) if sims[i].item() >= min_score]

    def clear(self):
        self.db.clear()
        self._emb_cache_tensor = None
        self._emb_cache_dirty = True
