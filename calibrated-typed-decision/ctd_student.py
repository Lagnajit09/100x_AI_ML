"""Calibrated typed-decision students, trained on m-lagnajit/calibrated-typed-decisions.

A student reads a text, a question and a list of options in one pass, and returns a calibrated chance for every
option. Each repository holds three students trained with different seeds; their dialed chances are averaged.

    student = CalibratedStudent.from_pretrained("m-lagnajit/ctd-student-f")
    student.predict_one(instructions="Which banking intent best describes this customer message?",
                        text="I still haven't received my new card, what should I do?",
                        options=["card arrival", "card delivery estimate", "lost or stolen card"])

The input layout, token limits and dials reproduce the training notebook exactly:
    [CLS] question [SEP] [MASK] option 1 [MASK] option 2 ... [SEP] text [SEP]
"""
import json
import os

import numpy as np
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

QUESTION_TYPES = {"choice": "choice", "yes/no": "noul", "noul": "noul"}


def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def option_texts(row):
    """What the student reads for each option, in stored order: 'label: description'; yes/no as 'no' and 'yes'."""
    if row["type"] == "noul":
        return ["no", "yes"]
    descs = row.get("option_descriptions") or [""] * len(row["options"])
    return [f"{o}: {d}" if d else o for o, d in zip(row["options"], descs)]


def apply_dial(scores, qtype, dial):
    """A temperature for every question; for yes/no, also a shift added to the 'yes' score."""
    T, b = dial
    z = np.asarray(scores, dtype=float) / T
    if qtype == "noul":
        z[1] += b
    return softmax(z)


class _Student(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.body = AutoModel.from_config(config)
        self.scorer = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, marker_pos, option_mask):
        h = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state   # (B, L, H)
        idx = marker_pos.unsqueeze(-1).expand(-1, -1, h.size(-1))                             # (B, K, H)
        scores = self.scorer(torch.gather(h, 1, idx)).squeeze(-1)                             # (B, K)
        return scores.masked_fill(~option_mask, float("-inf"))                                # empty slots


class CalibratedStudent:
    """Three seeds of one student, with their dials. Use from_pretrained(repo_id or local folder)."""

    def __init__(self, folder, device=None):
        from safetensors.torch import load_file
        with open(os.path.join(folder, "config.json")) as f:
            self.config = json.load(f)
        with open(os.path.join(folder, "dials.json")) as f:
            self.dials = {seed: {t: tuple(d) for t, d in v.items()} for seed, v in json.load(f).items()}
        c = self.config
        self.max_len, self.max_q, self.max_opt = c["max_len"], c["max_question_tokens"], c["max_option_tokens"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(c["base_model"])
        self.CLS, self.SEP = self.tok.cls_token_id, self.tok.sep_token_id
        self.MASK, self.PAD = self.tok.mask_token_id, self.tok.pad_token_id
        model_config = AutoConfig.from_pretrained(c["base_model"])
        if hasattr(model_config, "reference_compile"):
            model_config.reference_compile = False
        self.models = {}
        for seed in c["seeds"]:
            m = _Student(model_config)
            state = load_file(os.path.join(folder, f"{seed}.safetensors"))
            m.load_state_dict({k: v.float() for k, v in state.items()})     # stored in 16-bit, run in 32-bit
            self.models[seed] = m.float().to(self.device).eval()

    @classmethod
    def from_pretrained(cls, repo_or_folder, device=None):
        if os.path.isdir(repo_or_folder):
            return cls(repo_or_folder, device)
        from huggingface_hub import snapshot_download
        return cls(snapshot_download(repo_or_folder), device)

    # ---------- input, exactly as in training ----------
    def _tokenize(self, row):
        enc = lambda s: self.tok(s, add_special_tokens=False)["input_ids"]
        q = enc(row["instructions"])[:self.max_q]
        opts = [enc(" " + o)[:self.max_opt] for o in option_texts(row)]
        txt = enc(" " + row["state"])
        ids = [self.CLS] + q + [self.SEP]
        marker_pos = []
        for o in opts:                                  # stored order: no shuffling at prediction time
            marker_pos.append(len(ids))
            ids += [self.MASK] + o
        ids += [self.SEP]
        room = self.max_len - len(ids) - 1              # only the text can ever be cut
        return ids + txt[:room] + [self.SEP], marker_pos

    def _batch(self, rows):
        items = [self._tokenize(r) for r in rows]
        B, L, K = len(rows), max(len(i) for i, _ in items), max(len(m) for _, m in items)
        input_ids = torch.full((B, L), self.PAD, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        marker_pos = torch.zeros((B, K), dtype=torch.long)          # empty slots point at position 0
        option_mask = torch.zeros((B, K), dtype=torch.bool)
        for b, (ids, mp) in enumerate(items):
            input_ids[b, :len(ids)] = torch.tensor(ids)
            attn[b, :len(ids)] = 1
            marker_pos[b, :len(mp)] = torch.tensor(mp)
            option_mask[b, :len(mp)] = True
        return [t.to(self.device) for t in (input_ids, attn, marker_pos, option_mask)], [len(m) for _, m in items]

    @torch.no_grad()
    def raw_scores(self, rows, seed, batch_size=64):
        """Undialed scores of one seed, one array per row, in the options' stored order."""
        model, out = self.models[seed], []
        for i in range(0, len(rows), batch_size):
            feed, ks = self._batch(rows[i:i + batch_size])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                s = model(*feed).float().cpu().numpy()
            out += [s[b, :k] for b, k in enumerate(ks)]
        return out

    # ---------- predictions ----------
    def predict(self, rows, batch_size=64, return_seeds=False):
        """Calibrated chances for each row (a dict with 'instructions', 'state', 'options', 'type' and optionally
        'option_descriptions'): each seed's scores pass through its own dial, then the seeds are averaged."""
        rows = [dict(r, type=QUESTION_TYPES[r.get("type", "choice")]) for r in rows]
        for r in rows:
            if r["type"] == "noul":
                r["options"] = ["false", "true"]
        per_seed = {seed: [apply_dial(s, r["type"], self.dials[seed][r["type"]])
                           for s, r in zip(self.raw_scores(rows, seed, batch_size), rows)] for seed in self.models}
        ensemble = [np.mean([per_seed[seed][i] for seed in self.models], axis=0) for i in range(len(rows))]
        return (ensemble, per_seed) if return_seeds else ensemble

    def predict_one(self, instructions, text, options=None, descriptions=None, question_type="choice"):
        """One question. Choice: pass the options (and, ideally, a short description of each).
        Yes/no: put the candidate label in the question, and leave options empty."""
        row = {"instructions": instructions, "state": text, "type": question_type,
               "options": list(options or []), "option_descriptions": list(descriptions or [])}
        chances = self.predict([row])[0]
        names = ["no", "yes"] if QUESTION_TYPES[question_type] == "noul" else row["options"]
        return dict(zip(names, (round(float(p), 4) for p in chances)))