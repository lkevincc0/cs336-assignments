import os
import time
import pickle
from cs336_basics.bpe import train_bpe


def main():
    output_dir = "assignment1-basics/output"
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    vocab, merges = train_bpe(
        input_path="assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=256 + 1 + 10,
        special_tokens=["<|endoftext|>"],
    )
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    print(f"Final vocab size: {len(vocab)}")
    print("Sample vocab items:")
    for k, v in list(vocab.items())[:5]:
        print(f"  ID {k}: {v}")
    print("\nSave to pickle:")
    with open(f"{output_dir}/tiny_story_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{output_dir}/tiny_story_merges.pkl", "wb") as f:
        pickle.dump(merges, f)


if __name__ == "__main__":
    main()
