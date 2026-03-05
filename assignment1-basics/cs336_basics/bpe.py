import os
from multiprocessing import Pool
from typing import BinaryIO

import regex as re
from tqdm import tqdm


# Pretokenization
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_single_chunk(chunk_text: str, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_counts = {}

    # Handle special tokens
    if not special_tokens:
        text_parts = [chunk_text]
    else:
        # Escape and join specials
        escaped_specials = [re.escape(tok) for tok in special_tokens]
        split_pattern = "|".join(escaped_specials)

        # Split text by specials
        text_parts = re.split(split_pattern, chunk_text)

    # Tokenize each part
    for text_part in text_parts:
        if not text_part:
            continue

        for match in re.finditer(PAT, text_part):
            token_str = match.group(0)
            token_bytes = token_str.encode("utf-8")
            token_tuple = tuple(bytes([b]) for b in token_bytes)
            token_counts[token_tuple] = token_counts.get(token_tuple, 0) + 1

    return token_counts


def process_file_chunk(input_path: str, start: int, end: int, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")
    return pretokenize_single_chunk(chunk_text, special_tokens)


def get_global_token_counts(input_path: str, num_processes: int, special_tokens: list[str]) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        split_token_bytes = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        boundaries = find_chunk_boundaries(f, num_processes, split_token_bytes)
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens))
    global_token_counts = {}
    with Pool(processes=num_processes) as pool:
        async_results = [pool.apply_async(process_file_chunk, args=t) for t in tasks]
        for ar in tqdm(async_results, total=len(tasks), desc="Processing chunks"):
            chunk_counts = ar.get()
            for token_tuple, count in chunk_counts.items():
                global_token_counts[token_tuple] = global_token_counts.get(token_tuple, 0) + count
    return global_token_counts


def build_pair_indexes(token_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes, bytes], int]:
    pair_counts = {}
    for token_tuple, count in token_counts.items():
        for i in range(len(token_tuple) - 1):
            pair = (token_tuple[i], token_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    return pair_counts


# BPE Training
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    global_token_counts = get_global_token_counts(
        input_path, os.cpu_count() - 2 if os.cpu_count() > 2 else 1, special_tokens
    )

    vocab = {i: bytes([i]) for i in range(256)}  # Initial vocab of single bytes
    for i, spec in enumerate(special_tokens):
        vocab[256 + i] = spec.encode("utf-8")
    merges = []
    word_counts = global_token_counts
    pair_counts = {}
    word_by_pair = {}

    for word_tuple, count in word_counts.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            if pair not in word_by_pair:
                word_by_pair[pair] = set()
            word_by_pair[pair].add(word_tuple)

    num_merges = vocab_size - len(vocab)
    for i in range(num_merges):
        if not pair_counts:
            break

        # 找到频率最高的 pair
        best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], p))
        merges.append(best_pair)

        # 将合并后的新 token 加入词表
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # 获取包含最佳 best_pair 的所有单词序列，然后从缓存中移除最佳 best_pair
        words_to_process = word_by_pair.pop(best_pair)
        del pair_counts[best_pair]

        # 更新只受影响的单词序列
        for word in list(words_to_process):
            count = word_counts[word]
            for j in range(len(word) - 1):
                p = (word[j], word[j + 1])
                if p != best_pair:
                    pair_counts[p] -= count
                    if pair_counts[p] == 0:
                        del pair_counts[p]
                    if word in word_by_pair.get(p, set()):
                        word_by_pair[p].remove(word)
                        if not word_by_pair[p]:
                            del word_by_pair[p]

            # 把老单词从词典中删除
            del word_counts[word]

            # 构造发生了合并的新词序列
            new_word_list = []
            j = 0
            while j < len(word):
                if j < len(word) - 1 and word[j] == best_pair[0] and word[j + 1] == best_pair[1]:
                    new_word_list.append(new_token)
                    j += 2
                else:
                    new_word_list.append(word[j])
                    j += 1
            new_word = tuple(new_word_list)

            # 所有新的 pair 更新 pair_counts 和 反向索引
            word_counts[new_word] = count
            for j in range(len(new_word) - 1):
                p = (new_word[j], new_word[j + 1])
                pair_counts[p] = pair_counts.get(p, 0) + count
                if p not in word_by_pair:
                    word_by_pair[p] = set()
                word_by_pair[p].add(new_word)

    return vocab, merges


def test():
    input_path = "assignment1-basics/data/test.txt"
    special_tokens = ["<|endoftext|>"]
    num_processes = os.cpu_count() - 2 if os.cpu_count() > 2 else 1

    print(f"Starting pretokenization with {num_processes} processes...")
    global_token_counts = get_global_token_counts(input_path, num_processes, special_tokens)

    print("\n## Pretokenization")
    print(f"Total unique tokens: {len(global_token_counts)}")
    print("前5项：")
    for k, v in list(global_token_counts.items())[:5]:
        print(f"{k}: {v}")

    print("\n## Building Pair Indexes")
    pairs = build_pair_indexes(global_token_counts)
    print(f"Total unique pairs: {len(pairs)}")
    print("前5项：")
    for k, v in list(pairs.items())[:5]:
        print(f"{k}: {v}")

    print("\n## Training BPE")
    vocab_size = 256 + len(special_tokens) + 10  # 基础256 + 特殊字符1 + 10次Merge
    print(f"Target vocab_size: {vocab_size}")
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    print(f"\nFinal vocab size: {len(vocab)}")
    print("前5次 Merges:")
    for i, m in enumerate(merges[:5]):
        print(f"  Merge {i + 1}: {m}")

    print("\nVocab 最新加入的5项:")
    for k, v in list(vocab.items())[-5:]:
        print(f"  ID {k}: {v}")


# test()
