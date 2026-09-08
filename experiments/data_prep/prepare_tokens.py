"""Tokenize a text corpus with the GPT-2 BPE into nanoGPT-style uint16 .bin shards.

Writes {OUT}/train.bin and {OUT}/val.bin (uint16 token ids). Streams the HF dataset
so we never hold the whole corpus in memory, and caps the token budget with --max_train_tokens.

Requires: tiktoken, datasets  (uv add tiktoken datasets)

Example:
  python experiments/data_prep/prepare_tokens.py \
      --dataset openwebtext --out /n/holystore01/.../openwebtext_gpt2 \
      --max_train_tokens 2_000_000_000 --val_tokens 5_000_000
"""
import argparse, os, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="openwebtext",
                    help="HF dataset id: 'openwebtext' or e.g. 'HuggingFaceFW/fineweb-edu'")
    ap.add_argument("--config", default=None, help="HF dataset config/name (e.g. 'sample-10BT')")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_train_tokens", type=int, default=2_000_000_000)
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    args = ap.parse_args()

    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256, document separator
    os.makedirs(args.out, exist_ok=True)

    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)

    def write_split(path, budget):
        arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(budget,))
        n = 0
        for ex in ds:
            ids = enc.encode_ordinary(ex[args.text_key]); ids.append(eot)
            take = min(len(ids), budget - n)
            arr[n:n + take] = np.array(ids[:take], dtype=np.uint16)
            n += take
            if n >= budget:
                break
        arr.flush()
        print(f"  wrote {n:,} tokens -> {path}")
        return n

    # val first (small), then train (kept disjoint by consuming the stream in order)
    print("tokenizing val split...");   write_split(os.path.join(args.out, "val.bin"),   args.val_tokens)
    print("tokenizing train split..."); write_split(os.path.join(args.out, "train.bin"), args.max_train_tokens)
    print("done. vocab_size = 50304 (padded); GPT-2 BPE has 50257 real tokens.")

if __name__ == "__main__":
    main()
