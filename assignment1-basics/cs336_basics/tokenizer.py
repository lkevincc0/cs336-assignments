import pickle
import regex as re
from typing import Iterable, Iterator

# GPT-2 regex pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class PythonTokenizer:
    def __init__(
        self, vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id = {v: k for k, v in vocab.items()}

        # if special tokens not in vocab, add them
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes not in self.token_to_id:
                new_id = max(self.vocab.keys()) + 1 if self.vocab else 0
                self.vocab[new_id] = st_bytes
                self.token_to_id[st_bytes] = new_id

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        # cache mapping from original token string -> list[bytes] after BPE merging
        self._merge_cache: dict[str, list[bytes]] = {}
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        byte_seq = b"".join([self.vocab[id] for id in ids])
        return byte_seq.decode("utf-8")

    def _merge_word(self, word_bytes: list[bytes]) -> list[bytes]:
        while len(word_bytes) > 1:
            pairs = [(word_bytes[i], word_bytes[i+1]) for i in range(len(word_bytes)-1)]
            best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))
            if self.merge_ranks.get(best_pair, float('inf')) == float('inf'):
                break
            new_word_bytes: list[bytes] = []
            i = 0
            while i < len(word_bytes):
                if i < len(word_bytes) - 1 and (word_bytes[i], word_bytes[i+1]) == best_pair:
                    merged = word_bytes[i] + word_bytes[i+1]
                    new_word_bytes.append(merged)
                    i += 2
                else:
                    new_word_bytes.append(word_bytes[i])
                    i += 1
            word_bytes = new_word_bytes
        return word_bytes

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        if self.special_tokens:
            escaped_specials = [re.escape(tok) for tok in self.special_tokens]
            split_pattern = f"({'|'.join(escaped_specials)})"
            text_parts = re.split(split_pattern, text)
        else:
            text_parts = [text]
        for part in text_parts:
            if not part:
                continue
            if part in self.special_tokens:
                st_bytes = part.encode("utf-8")
                if st_bytes in self.token_to_id:
                    ids.append(self.token_to_id[st_bytes])
                continue
            for match in re.finditer(PAT, part):
                word_str = match.group(0)
                # use cache to avoid repeated merging for common words (Zipf's law)
                cached = self._merge_cache.get(word_str)
                if cached is not None:
                    merged_bytes = cached
                else:
                    word_bytes = [bytes([b]) for b in word_str.encode("utf-8")]
                    merged_bytes = self._merge_word(word_bytes)
                    # store a copy in cache
                    self._merge_cache[word_str] = list(merged_bytes)
                for b in merged_bytes:
                    if b in self.token_to_id:
                        ids.append(self.token_to_id[b])
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for t in iterable:
            for ti in self.encode(t):
                yield ti


try:
    from fast_cs336 import Tokenizer as _RustTokenizer  # type: ignore
    Tokenizer = _RustTokenizer
except Exception:
    Tokenizer = PythonTokenizer