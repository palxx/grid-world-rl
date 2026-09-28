"""Offline tokenizer stub that mimics Qwen2/3's pre-tokenizer (single digits, ' WORD' as one token,
punctuation runs absorbing trailing newlines). Each pre-token is assumed to be one BPE token, which is
optimistic for long words; scripts/check_tokenizer.py against the real tokenizer is the authoritative check."""
import regex

_PAT = regex.compile(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")


class StubTokenizer:
    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.setdefault(t, len(self.vocab)) for t in _PAT.findall(text)]
