"""
bn_grapheme.py
──────────────
Bengali grapheme tokenizer for OCR.

Bengali script uses grapheme clusters (base consonant + optional vowel
diacritics / hasanta / conjuncts). We tokenize at the Unicode grapheme
cluster level so CTC works on meaningful visual units.

Unicode ranges used:
  0980–09FF  Bengali block
  200C        Zero Width Non-Joiner (ZWNJ)
  200D        Zero Width Joiner (ZWJ)

Special tokens
──────────────
  PAD = 0   (also used as CTC blank)
  UNK = 1
  BOS = 2   (begin of sequence — not used in CTC but kept for compatibility)
  EOS = 3   (end of sequence — not used in CTC but kept for compatibility)
"""

import unicodedata
import regex  # pip install regex  (handles Unicode grapheme clusters)
from typing import List

# ── Bengali Unicode ranges ────────────────────────────────────────────────────
# We build a fixed vocabulary from all printable Bengali characters + digits
# + punctuation commonly found in BanglaWriting.

_BENGALI_CHARS = []

# Bengali block: U+0980 – U+09FF
for cp in range(0x0980, 0x0A00):
    ch = chr(cp)
    cat = unicodedata.category(ch)
    # Keep letters (L*), marks (M*), digits (Nd), punctuation (P*)
    if cat.startswith(('L', 'M', 'N', 'P')) or ch in ('\u200C', '\u200D'):
        _BENGALI_CHARS.append(ch)

# ASCII digits and common punctuation (for mixed documents)
for ch in '0123456789।॥-–—,.!?':
    if ch not in _BENGALI_CHARS:
        _BENGALI_CHARS.append(ch)

# Sort for determinism
_BENGALI_CHARS = sorted(set(_BENGALI_CHARS))


class BnGraphemeTokenizer:
    """
    Maps Bengali grapheme clusters ↔ integer IDs.

    Special IDs
    ───────────
    0 → PAD / CTC blank
    1 → UNK
    2 → BOS
    3 → EOS
    4+ → actual grapheme clusters
    """

    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3
    _RESERVED = 4

    def __init__(self):
        # Build char→id and id→char tables
        # We enumerate all single Bengali characters first, then common
        # multi-character grapheme clusters (conjuncts) are handled by
        # the regex grapheme splitter at encode/decode time.

        self._char2id: dict = {}
        self._id2char: dict = {
            self.PAD: '<PAD>',
            self.UNK: '<UNK>',
            self.BOS: '<BOS>',
            self.EOS: '<EOS>',
        }

        idx = self._RESERVED
        for ch in _BENGALI_CHARS:
            if ch not in self._char2id:
                self._char2id[ch] = idx
                self._id2char[idx] = ch
                idx += 1

        self._vocab_size = idx

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # ── Grapheme splitting ────────────────────────────────────────────────────

    @staticmethod
    def _split_graphemes(text: str) -> List[str]:
        """
        Split a Bengali string into grapheme clusters.

        Uses the `regex` library which implements Unicode TR#29
        grapheme cluster boundaries. This correctly handles:
          - consonant + hasanta (virama) + consonant conjuncts
          - vowel diacritics attached to base consonants
          - independent vowels
        """
        try:
            # \X matches a single Unicode extended grapheme cluster
            return regex.findall(r'\X', text)
        except Exception:
            # Fallback: character-by-character
            return list(text)

    # ── Encode / Decode ───────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        """
        text (Bengali string) → list of int token IDs.

        Unknown graphemes map to UNK.
        PAD / BOS / EOS are NOT added here; the training loop handles them.
        """
        if not text:
            return []
        ids = []
        for g in self._split_graphemes(text):
            # Try whole grapheme first (e.g., conjunct stored as multi-char)
            if g in self._char2id:
                ids.append(self._char2id[g])
            else:
                # Fall back to character-by-character inside the grapheme
                mapped = False
                for ch in g:
                    if ch in self._char2id:
                        ids.append(self._char2id[ch])
                        mapped = True
                if not mapped:
                    ids.append(self.UNK)
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        list of int token IDs → Bengali string.

        Special tokens (PAD, UNK, BOS, EOS) are dropped.
        """
        chars = []
        skip = {self.PAD, self.BOS, self.EOS}
        for i in ids:
            if i in skip:
                continue
            if i == self.UNK:
                chars.append('?')
            else:
                chars.append(self._id2char.get(i, '?'))
        return ''.join(chars)

    def __repr__(self):
        return (f"BnGraphemeTokenizer("
                f"vocab_size={self.vocab_size}, "
                f"PAD={self.PAD}, UNK={self.UNK})")


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    tok = BnGraphemeTokenizer()
    print(tok)

    samples = [
        'বাংলা',      # "Bangla"
        'আমার সোনার বাংলা',   # national anthem start
        'কলম',        # pen
        '১২৩',        # Bengali digits
    ]
    for s in samples:
        ids = tok.encode(s)
        rec = tok.decode(ids)
        ok  = '✓' if rec == s else '✗'
        print(f"  {ok}  {s!r:25s}  ids={ids}  decoded={rec!r}")