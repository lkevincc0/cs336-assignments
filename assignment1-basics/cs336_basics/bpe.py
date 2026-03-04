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


def test_pretokenization():
    input_path = "assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    num_processes = os.cpu_count() - 2

    print(f"Starting pretokenization with {num_processes} processes...")
    global_token_counts = get_global_token_counts(input_path, num_processes, special_tokens)

    print(f"Total unique tokens: {len(global_token_counts)}")
    print("前5项：")
    for k, v in list(global_token_counts.items())[:5]:
        print(f"{k}: {v}")


# BPE Training
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    # return vocab, merges
    pass


# test_pretokenization()
