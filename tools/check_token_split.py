#!/usr/bin/env python
"""Assert that the token .bin splits in a directory are genuinely disjoint (C-D2 / F4).

Three checks per pair of splits:
  * prefix    -- the shorter file is not a prefix of the longer one (the F4 bug:
                 `np.array_equal(train[:len(val)], val)` was True for fineweb_edu_gpt2);
  * documents -- (near-)exhaustive: every complete eot-terminated document of every
                 split is hashed, so a shared document anywhere is *seen* (the block
                 sample below only covers ~0.1% of a large file).  A shared fraction
                 above --max_shared_doc_frac of the smaller split fails.  The tolerance
                 is not zero because FineWeb-edu itself contains exact duplicate
                 documents at stream positions far apart (~1.5% of documents have a twin
                 somewhere in the 10BT sample; measured 2026-09-18 on the v2 bins:
                 3/7,729 val and 4/7,932 test documents also occur in the 250M-token
                 train split, i.e. 0.04%, and none occurs twice inside its own split).
                 The F4 bug, by contrast, shared 100% of them;
  * blocks    -- a rolling sample of fixed-length blocks taken at evenly spaced
                 offsets in one split does not occur verbatim anywhere in the other.
                 Kept because it also catches overlaps that do not align to documents.
A trailing run of more than --max_zero_tail zero tokens (an unfilled file tail left by
a stream that ran dry) is also a failure.

Usage:
  python tools/check_token_split.py $SV3_DATA_ROOT/fineweb_edu_gpt2_v2
  python tools/check_token_split.py DIR --sample_blocks 32 --block_len 256
Exit status 0 = disjoint, 1 = overlap found (or nothing to check).
"""
import argparse, hashlib, json, os, sys
import numpy as np

SPLITS = ("train", "val", "test")


def load_splits(directory):
    """{name: memmap} for every {name}.bin that exists in `directory`."""
    out = {}
    for name in SPLITS:
        path = os.path.join(directory, name + ".bin")
        if os.path.isfile(path):
            out[name] = np.memmap(path, dtype=np.uint16, mode="r")
    return out


def is_prefix(a, b):
    """True if the shorter of `a`, `b` is a prefix of the other."""
    n = min(len(a), len(b))
    return bool(n) and bool(np.array_equal(np.asarray(a[:n]), np.asarray(b[:n])))


def block_offsets(n_tokens, n_blocks, block_len):
    """Evenly spaced start offsets for a rolling sample of blocks."""
    last = n_tokens - block_len
    if last < 0 or n_blocks < 1:
        return []
    if n_blocks == 1:
        return [0]
    step = last / (n_blocks - 1)
    return sorted({int(round(i * step)) for i in range(n_blocks)})


def find_shared_blocks(a, b, n_blocks=32, block_len=256):
    """Offsets (in a, in b) of sampled blocks of `a` that occur verbatim in `b`.

    Per block: one pass over `b` collects the positions matching its first token, then
    the candidate set is refined one token at a time.  Everything stays vectorised --
    a Python loop over the candidates is hopeless here, because a common anchor token
    matches millions of positions in a 250M-token train.bin.
    """
    offs = block_offsets(len(a), n_blocks, block_len)
    if not offs or len(b) < block_len:
        return []
    b_arr = np.asarray(b)
    limit = len(b_arr) - block_len + 1
    hits = []
    for o in offs:
        blk = np.asarray(a[o:o + block_len])
        cand = np.flatnonzero(b_arr[:limit] == blk[0])
        for j in range(1, block_len):
            if cand.size == 0:
                break
            cand = cand[b_arr[cand + j] == blk[j]]     # a few tokens empty it
        hits.extend((int(o), int(pos)) for pos in cand)
    return hits


def document_hashes(a, eot=50256, min_tokens=8):
    """Digests of every complete ``eot``-terminated document in ``a``.

    Splitting on the document separator makes the comparison exhaustive and cheap: two
    splits share data iff their digest sets intersect (a truncated document -- the tail
    the budget cut off -- has no terminator and is skipped, as is any document shorter
    than ``min_tokens``, which would collide by chance).
    """
    arr = np.asarray(a)
    ends = np.flatnonzero(arr == eot)
    out = set()
    start = 0
    for end in ends.tolist():
        if end - start >= min_tokens:
            out.add(hashlib.blake2b(arr[start:end].tobytes(), digest_size=16).digest())
        start = end + 1
    return out


def shared_doc_fraction(n_shared, n_docs_a, n_docs_b):
    """Shared documents as a fraction of the smaller split's document count."""
    smaller = min(n_docs_a, n_docs_b)
    return (n_shared / smaller) if smaller else 0.0


def trailing_zeros(a):
    """Length of the trailing run of zero tokens (an unfilled file tail)."""
    tail = np.asarray(a[-min(len(a), 1_000_000):])
    nz = np.flatnonzero(tail)
    return int(len(tail) - nz[-1] - 1) if len(nz) else int(len(tail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="directory holding train.bin / val.bin / test.bin")
    ap.add_argument("--sample_blocks", type=int, default=32)
    ap.add_argument("--block_len", type=int, default=256)
    ap.add_argument("--eot", type=int, default=50256, help="document separator token")
    ap.add_argument("--max_shared_doc_frac", type=float, default=0.002,
                    help="fail when more than this fraction of the smaller split's "
                         "documents also occur in the other split (default 0.2%%; the "
                         "corpus itself has ~0.04%% residual duplicates)")
    ap.add_argument("--max_zero_tail", type=int, default=1024,
                    help="a longer trailing run of zero tokens is a failure")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # progress visible in a batch log

    splits = load_splits(args.directory)
    if not splits:
        print(f"no {'/'.join(s + '.bin' for s in SPLITS)} in {args.directory}")
        return 1

    ok = True
    meta_path = os.path.join(args.directory, "token_counts.json")
    print(f"directory: {args.directory}")
    for name, arr in splits.items():
        zeros = trailing_zeros(arr)
        print(f"  {name:>5}.bin  {len(arr):>13,} tokens  {len(arr) * 2:>14,} bytes"
              f"  trailing zeros: {zeros}")
        if zeros > args.max_zero_tail:
            print(f"  FAIL {name}.bin ends in {zeros:,} zero tokens (unfilled tail; the "
                  "file was not truncated to the tokens actually written)")
            ok = False
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  token_counts.json: {json.dumps(meta.get('tokens', meta))}")
        for name, arr in splits.items():
            recorded = meta.get("tokens", {}).get(name)
            if recorded is not None and recorded != len(arr):
                print(f"  FAIL {name}.bin holds {len(arr):,} tokens, json records {recorded:,}")
                return 1

    docs = {}
    for name, arr in splits.items():
        docs[name] = document_hashes(arr, args.eot)
        print(f"  {name:>5}.bin  {len(docs[name]):>13,} complete documents", flush=True)

    names = list(splits)
    shared_total = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = docs[a] & docs[b]
            shared_total += len(shared)
            frac = shared_doc_fraction(len(shared), len(docs[a]), len(docs[b]))
            if frac > args.max_shared_doc_frac:
                print(f"  FAIL documents: {len(shared):,} document(s) occur in both "
                      f"{a} and {b} ({frac:.3%} of the smaller split, tolerance "
                      f"{args.max_shared_doc_frac:.3%})")
                ok = False
            elif shared:
                print(f"  ok   documents: {a} and {b} share {len(shared):,} document(s) "
                      f"({frac:.3%} of the smaller split; residual exact duplicates of "
                      "the corpus, under the tolerance)")
            else:
                print(f"  ok   documents: {a} and {b} share no document")
            if is_prefix(splits[a], splits[b]):
                print(f"  FAIL prefix: {a} and {b} share a common prefix of "
                      f"{min(len(splits[a]), len(splits[b])):,} tokens")
                ok = False
            else:
                print(f"  ok   prefix: {a} vs {b}")
            for src, dst in ((a, b), (b, a)):
                hits = find_shared_blocks(splits[src], splits[dst],
                                          args.sample_blocks, args.block_len)
                if hits:
                    print(f"  FAIL blocks: {len(hits)} of the sampled {args.block_len}-token "
                          f"blocks of {src} occur in {dst} (first: {hits[0]})")
                    ok = False
                else:
                    print(f"  ok   blocks: no sampled block of {src} occurs in {dst}")
    if not ok:
        print("OVERLAP DETECTED")
    elif shared_total:
        print(f"DISJOINT (up to {shared_total} residual duplicate documents of the corpus, "
              f"under the {args.max_shared_doc_frac:.3%} tolerance)")
    else:
        print("DISJOINT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
