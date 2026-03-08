import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer

start_time = time.time()

tokenizer = Tokenizer.from_files(
    vocab_filepath="assignment1-basics/output/tiny_story_vocab.pkl",
    merges_filepath="assignment1-basics/output/tiny_story_merges.pkl",
    special_tokens=["<|endoftext|>"],
)

input_file = "assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"

print(f"Encoding {input_file} ...")
token_ids = tokenizer.encode_file(input_file)

token_ids_np = np.array(token_ids, dtype=np.uint16)
np.save("assignment1-basics/output/tinystories_train_tokens.npy", token_ids_np)
print(f"Done, took {time.time() - start_time:.2f} seconds. {len(token_ids_np)} tokens saved.")
