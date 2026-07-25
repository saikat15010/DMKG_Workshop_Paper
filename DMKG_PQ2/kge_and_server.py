# kge_and_server.py — K-silo TransE training + federated fusion server
#
# Same model class and training math as FedV-KGQA (transe_model.py, server.py),
# generalized from a fixed 3 silos to a list of K silos so the experiment
# driver can instantiate it for K in {3, 5, 7}.

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from config import (BERT_MODEL, MLP_HIDDEN_DIMS, MLP_DROPOUT, KGE_EMBED_DIM,
                     KGE_NORM, KGE_LR, KGE_MARGIN, KGE_BATCH_SIZE,
                     KGE_NEG_SAMPLES, USE_TOPIC_ANCHORING)


# ── TransE (unchanged from FedV-KGQA transe_model.py) ────────────────────────

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, norm=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm = norm
        self.ent_embed = nn.Embedding(num_entities, embed_dim)
        self.rel_embed = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.ent_embed.weight)
        nn.init.xavier_uniform_(self.rel_embed.weight)
        with torch.no_grad():
            self.ent_embed.weight.data = F.normalize(self.ent_embed.weight.data, p=2, dim=-1)

    def score(self, h_ids, r_ids, t_ids):
        h, r, t = self.ent_embed(h_ids), self.rel_embed(r_ids), self.ent_embed(t_ids)
        return -torch.norm(h + r - t, p=self.norm, dim=-1)

    def margin_ranking_loss(self, h_ids, r_ids, t_pos_ids, t_neg_ids, margin=1.0):
        pos_scores = self.score(h_ids, r_ids, t_pos_ids)
        B, K = t_neg_ids.shape
        h_exp = h_ids.unsqueeze(1).expand(B, K).reshape(-1)
        r_exp = r_ids.unsqueeze(1).expand(B, K).reshape(-1)
        t_neg_flat = t_neg_ids.reshape(-1)
        neg_scores = self.score(h_exp, r_exp, t_neg_flat).reshape(B, K)
        hard_neg_scores, _ = neg_scores.max(dim=1)
        return F.relu(margin - pos_scores + hard_neg_scores).mean()

    def get_entity_embeddings(self):
        return self.ent_embed.weight


class KGETripleDataset(Dataset):
    def __init__(self, silo_triples, shared_entity2id, relation2id):
        self.data = []
        for h, r, t in silo_triples:
            h_id, r_id, t_id = shared_entity2id.get(h), relation2id.get(r), shared_entity2id.get(t)
            if h_id is not None and r_id is not None and t_id is not None:
                self.data.append((h_id, r_id, t_id))

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        h, r, t = self.data[idx]
        return torch.tensor(h), torch.tensor(r), torch.tensor(t)


def train_one_silo_kge(silo_triples, shared_entity2id, device, epochs, log=None):
    """Trains a local TransE model on one silo's triples. Returns (model, elapsed_sec)."""
    relations = sorted({r for _, r, _ in silo_triples})
    relation2id = {r: i for i, r in enumerate(relations)}
    dataset = KGETripleDataset(silo_triples, shared_entity2id, relation2id)

    num_entities = len(shared_entity2id)
    model = TransE(num_entities, max(len(relation2id), 1), KGE_EMBED_DIM, norm=KGE_NORM).to(device)

    if len(dataset) == 0:
        # empty silo (can happen at high K with few relations left) — leave
        # embeddings at their random init; it contributes no signal, which
        # is itself part of the imbalance cost the partition pays.
        return model, 0.0

    loader = DataLoader(dataset, batch_size=min(KGE_BATCH_SIZE, len(dataset)),
                         shuffle=True, num_workers=0)
    optimizer = optim.Adam(model.parameters(), lr=KGE_LR)

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for h, r, t_pos in loader:
            h, r, t_pos = h.to(device), r.to(device), t_pos.to(device)
            t_neg = torch.randint(0, num_entities, (h.size(0), KGE_NEG_SAMPLES), device=device)
            optimizer.zero_grad()
            loss = model.margin_ranking_loss(h, r, t_pos, t_neg, margin=KGE_MARGIN)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.ent_embed.weight.data = F.normalize(model.ent_embed.weight.data, p=2, dim=-1)
        if log and (epoch % 20 == 0 or epoch == epochs):
            log(f"    [KGE] epoch {epoch}/{epochs}  loss={loss.item():.4f}")
    return model, time.time() - start


# ── Federated server, generalized to K silos ─────────────────────────────────

class QuestionEncoder(nn.Module):
    def __init__(self, embed_dim, K, hidden_dims=None, dropout=0.1):
        super().__init__()
        output_dim = K * embed_dim
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for p in self.bert.parameters():
            p.requires_grad = False
        hidden_dims = hidden_dims or MLP_HIDDEN_DIMS
        layers, in_dim = [], 768
        for h_dim in hidden_dims:
            layers += [nn.Linear(in_dim, h_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h_dim
        layers += [nn.Linear(in_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, questions, device):
        enc = self.tokenizer(questions, return_tensors="pt", padding=True,
                              truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = self.bert(**enc)
        cls = out.last_hidden_state[:, 0, :]
        return self.mlp(cls)


class FedServer(nn.Module):
    """K-silo generalization of FedV-KGQA's FedVServer (server.py)."""

    def __init__(self, embed_dim, K):
        super().__init__()
        self.embed_dim = embed_dim
        self.K = K
        self.question_encoder = QuestionEncoder(embed_dim, K)

    def fuse(self, silo_embeddings):
        """silo_embeddings: list of K tensors (N, d) -> (N, K*d)."""
        return torch.cat(silo_embeddings, dim=-1)

    def score_candidates(self, q_embed, h_joint, candidate_ids):
        safe_ids = candidate_ids.clamp(min=0)
        h_cands = h_joint[safe_ids]
        q_norm = F.normalize(q_embed, p=2, dim=-1).unsqueeze(1)
        h_norm = F.normalize(h_cands, p=2, dim=-1)
        return (q_norm * h_norm).sum(dim=-1)

    def ranking_loss(self, sim, answer_ids_batch, candidate_ids, margin=1.0):
        device = sim.device
        losses = []
        for i, answer_ids in enumerate(answer_ids_batch):
            cands = candidate_ids[i]
            valid_mask = cands >= 0
            cand_list = cands[valid_mask].tolist()
            scores = sim[i][valid_mask]
            answer_set = set(answer_ids)
            pos_indices = [j for j, c in enumerate(cand_list) if c in answer_set]
            if not pos_indices:
                continue
            best_pos = scores[pos_indices].max()
            neg_mask = torch.ones(len(cand_list), dtype=torch.bool, device=device)
            for j in pos_indices:
                neg_mask[j] = False
            if neg_mask.sum() == 0:
                continue
            hard_neg = scores[neg_mask].max()
            losses.append(F.relu(margin + hard_neg - best_pos))
        if not losses:
            return torch.tensor(0.0, requires_grad=True, device=device)
        return torch.stack(losses).mean()

    def forward(self, questions, topic_ids, silo_embeddings, answer_ids_batch,
                candidate_ids, device, margin=1.0):
        h_joint = self.fuse(silo_embeddings)
        q_embed = self.question_encoder(questions, device)
        if USE_TOPIC_ANCHORING:
            q_final = q_embed + h_joint[topic_ids]
        else:
            q_final = q_embed
        candidate_ids = candidate_ids.to(device)
        sim = self.score_candidates(q_final, h_joint, candidate_ids)
        loss = self.ranking_loss(sim, answer_ids_batch, candidate_ids, margin)
        return loss, sim
