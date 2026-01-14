from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from pathlib import Path
import json

# We use CTC, so include a BLANK token id=0
BLANK = "<blank>"
PAD = "<pad>"
UNK = "<unk>"

@dataclass
class CharTokenizer:
    char2id: Dict[str, int]
    id2char: Dict[int, str]
    blank_id: int = 0
    pad_id: int = 1
    unk_id: int = 2

    @classmethod
    def build_from_texts(cls, texts: List[str]) -> "CharTokenizer":
        chars = set()
        for t in texts:
            for ch in t:
                chars.add(ch)
        chars = sorted(chars)

        # reserve: 0 blank, 1 pad, 2 unk
        char2id = {BLANK: 0, PAD: 1, UNK: 2}
        for ch in chars:
            if ch in char2id:
                continue
            char2id[ch] = len(char2id)

        id2char = {i: c for c, i in char2id.items()}
        return cls(char2id=char2id, id2char=id2char)

    def encode(self, text: str) -> List[int]:
        out = []
        for ch in text:
            out.append(self.char2id.get(ch, self.unk_id))
        return out

    def decode(self, ids: List[int]) -> str:
        # For CTC-decoded sequences, blanks are usually removed already.
        chars = []
        for i in ids:
            if i in (self.blank_id, self.pad_id):
                continue
            chars.append(self.id2char.get(i, ""))
        return "".join(chars)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                {"char2id": self.char2id, "blank_id": self.blank_id, "pad_id": self.pad_id, "unk_id": self.unk_id},
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        char2id = obj["char2id"]
        id2char = {int(v): k for k, v in char2id.items()}
        return cls(
            char2id=char2id,
            id2char=id2char,
            blank_id=obj["blank_id"],
            pad_id=obj["pad_id"],
            unk_id=obj["unk_id"],
        )
