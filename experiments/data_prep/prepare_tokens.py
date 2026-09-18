"""Tokenize a text corpus with the GPT-2 BPE into nanoGPT-style uint16 .bin shards.

Writes {OUT}/val.bin, {OUT}/test.bin and {OUT}/train.bin (uint16 token ids) plus
{OUT}/token_counts.json. The three splits are drawn from **disjoint documents**: one
single iterator over the streaming dataset is shared by all three writers, in the order
val -> test -> train, so a document that is consumed by one split is never seen by the
next (the tail of the document that fills a budget is discarded, not carried over).
Each file is truncated to the number of tokens actually written, so no split ends in a
run of zeros even if the stream runs dry.

Existing .bin files are NEVER overwritten without --force.

Requires: tiktoken, datasets  (uv add tiktoken datasets)

Example (as used for fineweb_edu_gpt2_v2):
  python experiments/data_prep/prepare_tokens.py \
      --dataset HuggingFaceFW/fineweb-edu --config sample-10BT \
      --out /n/holystore01/.../datasets/fineweb_edu_gpt2_v2 \
      --max_train_tokens 250_000_000 --val_tokens 8_000_000 --test_tokens 8_000_000
"""
import argparse, json, os, sys
import numpy as np

SPLITS = ("val", "test", "train")      # the order the shared iterator is consumed in


def existing_bins(out_dir):
    """Names of the splits that already have a .bin in ``out_dir`` (overwrite guard)."""
    return [s for s in SPLITS if os.path.isfile(os.path.join(out_dir, s + ".bin"))]


def write_split(texts, path, budget, encode, eot):
    """Write up to ``budget`` uint16 tokens from ``texts`` into ``path``.

    ``texts`` is ONE shared iterator over document strings, consumed here and resumed by
    the next call, so no document reaches two splits (`for ex in ds` would restart the
    stream and make val a prefix of train -- F4).  The document that fills the budget is
    truncated and its tail discarded.  The file is truncated to exactly the tokens
    written (2 bytes each), so there is no zero tail; a stream that runs dry gives a
    short-but-correct file and a loud warning, and an empty split raises.

    Returns the number of tokens written.
    """
    if budget < 1:
        raise ValueError(f"{path}: token budget must be >= 1, got {budget}")
    arr = np.memmap(path, dtype=np.uint16, mode="w+", shape=(budget,))
    n = 0
    for text in texts:
        ids = encode(text); ids.append(eot)
        take = min(len(ids), budget - n)
        arr[n:n + take] = np.array(ids[:take], dtype=np.uint16)
        n += take
        if n >= budget:
            break
    arr.flush()
    del arr                      # release the mapping before resizing the file
    os.truncate(path, n * 2)     # uint16 -> 2 bytes per token; no zero tail
    if n == 0:
        raise ValueError(f"{path}: the document stream ran dry before this split got any "
                         "tokens (an earlier split consumed the whole corpus)")
    if n < budget:
        print(f"  WARNING {path}: stream ran dry, only {n:,} of the {budget:,} "
              "requested tokens", flush=True)
    print(f"  wrote {n:,} tokens -> {path}", flush=True)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="openwebtext",
                    help="HF dataset id: 'openwebtext' or e.g. 'HuggingFaceFW/fineweb-edu'")
    ap.add_argument("--config", default=None, help="HF dataset config/name (e.g. 'sample-10BT')")
    ap.add_argument("--text_key", default="text")
    ap.add_argument("--out", required=True, help="output directory for the .bin files")
    ap.add_argument("--max_train_tokens", type=int, default=2_000_000_000)
    ap.add_argument("--val_tokens", type=int, default=5_000_000)
    ap.add_argument("--test_tokens", type=int, default=5_000_000)
    ap.add_argument("--force", action="store_true",
                    help="overwrite .bin files that already exist in --out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    present = existing_bins(args.out)
    if present and not args.force:
        sys.exit(f"refusing to overwrite {present} in {args.out}; pass --force "
                 "(NEVER point this at the verified token directories)")

    import tiktoken
    from datasets import load_dataset
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256, document separator

    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    # ONE iterator, shared by every split.
    texts = (ex[args.text_key] for ex in iter(ds))

    budgets = {"val": args.val_tokens, "test": args.test_tokens,
               "train": args.max_train_tokens}
    counts = {}
    for name in SPLITS:
        print(f"tokenizing {name} split...", flush=True)
        counts[name] = write_split(texts, os.path.join(args.out, f"{name}.bin"),
                                   budgets[name], enc.encode_ordinary, eot)

    meta = {"dataset": args.dataset, "config": args.config, "text_key": args.text_key,
            "order": list(SPLITS), "eot_token": int(eot),
            "vocab_size": 50304, "dtype": "uint16", "tokens": counts}
    with open(os.path.join(args.out, "token_counts.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done. vocab_size = 50304 (padded); GPT-2 BPE has 50257 real tokens.")
    print(json.dumps(counts))
    # The HF streaming reader keeps background threads alive; letting the interpreter
    # finalize with them running produced a fatal PyGILState_Release on the previous run
    # (datasets/tokenize.log). Everything is flushed above, so leave immediately.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
