# qa_dataset.py — QA torch Dataset + collate fn (from FedV-KGQA dataset.py, unchanged logic)

import torch
from torch.utils.data import Dataset
from data_prep import parse_qa_file


class QADataset(Dataset):
    def __init__(self, qa_path, shared_entity2id):
        self.samples = []
        for question, topic_entity, answers in parse_qa_file(qa_path):
            topic_id = shared_entity2id.get(topic_entity, -1)
            answer_ids = [shared_entity2id[a] for a in answers if a in shared_entity2id]
            if answer_ids and topic_id != -1:
                self.samples.append((question, topic_id, answer_ids))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx): return self.samples[idx]


def make_collate_fn(candidate_map):
    def collate_fn(batch):
        questions = [item[0] for item in batch]
        topic_ids = torch.tensor([item[1] for item in batch], dtype=torch.long)
        answer_ids = [item[2] for item in batch]
        cand_tensors = [torch.tensor(sorted(candidate_map.get(item[1], {item[1]})),
                                      dtype=torch.long) for item in batch]
        max_k = max(t.shape[0] for t in cand_tensors)
        padded = torch.full((len(batch), max_k), -1, dtype=torch.long)
        for i, t in enumerate(cand_tensors):
            padded[i, :t.shape[0]] = t
        return questions, topic_ids, answer_ids, padded
    return collate_fn


def hits_at_3(server, silo_models, qa_loader, device):
    """Lightweight Hits@3 check used only to detect the training round T at
    which the quality target is reached (drives M1's 'communication to
    target'). Full MRR/Hits@1/3/5/10 evaluation should still use your
    existing evaluate.py for the answer-quality control table."""
    server.eval()
    for m in silo_models: m.eval()
    hits, total = 0, 0
    with torch.no_grad():
        h_list = [m.get_entity_embeddings().to(device) for m in silo_models]
        h_joint = server.fuse(h_list)
        for questions, topic_ids, answer_ids_batch, candidate_ids in qa_loader:
            topic_ids = topic_ids.to(device)
            candidate_ids = candidate_ids.to(device)
            q_embed = server.question_encoder(questions, device)
            q_final = q_embed + h_joint[topic_ids]
            sim = server.score_candidates(q_final, h_joint, candidate_ids)
            for i, answer_ids in enumerate(answer_ids_batch):
                cands = candidate_ids[i]
                valid = cands >= 0
                cand_list = cands[valid].tolist()
                scores = sim[i][valid].tolist()
                ranked = sorted(zip(cand_list, scores), key=lambda x: x[1], reverse=True)[:3]
                if any(eid in set(answer_ids) for eid, _ in ranked):
                    hits += 1
                total += 1
    return hits / max(total, 1)
