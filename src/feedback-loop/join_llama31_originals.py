"""
Builds ragtruth-originals-*.jsonl for the llama-3.1 controlled dataset.

Unlike join_ragtruth_originals.py, the answers here are NOT in RAGTruth's
response.jsonl — they were generated fresh by Llama-3.1-8B and live in
ragtruth_llama31_annotated.csv. Only the question/passages come from RAGTruth's
source_info.jsonl (bundled inside the archive, so no GitHub download needed).

Mapping: response_id "llama31_8b_12167" -> source_id "12167" (verified against
metrics.csv, which carries both columns explicitly).

Usage:
  python3 join_llama31_originals.py \
      --archive-root hallu-llama31-controlled-r4-20260830 \
      --out ragtruth-originals-llama31.jsonl
"""
import json
import csv
import glob
import os
import argparse
import sys


def find_inputs_dir(root):
    hits = glob.glob(os.path.join(
        root, "data/inputs/inputs/llama31-controlled/*/ragtruth_llama31_annotated.csv"))
    if not hits:
        raise SystemExit(f"ragtruth_llama31_annotated.csv not found under {root}")
    return os.path.dirname(hits[0])


def find_run_dir(root):
    hits = glob.glob(os.path.join(root, "data/runs/*/support-critical/metrics.csv"))
    if not hits:
        raise SystemExit(f"support-critical/metrics.csv not found under {root}")
    return os.path.dirname(hits[0])


def load_sources(path):
    """source_id -> {question, passages} for QA rows only."""
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_type") != "QA":
                continue
            si = d["source_info"]
            if not isinstance(si, dict):
                continue
            idx[d["source_id"]] = {
                "question": si.get("question", ""),
                "context": si.get("passages", ""),
            }
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-root", required=True,
                    help="unpacked hallu-llama31-controlled-r4-* directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    inputs_dir = find_inputs_dir(args.archive_root)
    run_dir = find_run_dir(args.archive_root)
    print(f"inputs: {inputs_dir}", file=sys.stderr)
    print(f"run:    {run_dir}", file=sys.stderr)

    sources = load_sources(os.path.join(inputs_dir, "source_info.jsonl"))

    # metrics.csv is the authority for the response_id -> source_id link,
    # and also carries split / y / gen_model.
    meta = {}
    with open(os.path.join(run_dir, "metrics.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            meta[row["response_id"]] = {
                "source_id": row["source_id"],
                "split": row["split"],
                "y": row["y"],
                "gen_model": row["gen_model"],
            }

    n_ok = n_no_meta = n_no_src = 0
    with open(os.path.join(inputs_dir, "ragtruth_llama31_annotated.csv"),
              encoding="utf-8") as f, \
         open(args.out, "w", encoding="utf-8") as out:
        for row in csv.DictReader(f):
            rid = row["id"]
            m = meta.get(rid)
            if m is None:
                # fall back to stripping the prefix if the row isn't scored
                sid = rid.rsplit("_", 1)[-1]
                n_no_meta += 1
            else:
                sid = m["source_id"]

            src = sources.get(sid)
            if src is None:
                n_no_src += 1
                continue

            out.write(json.dumps({
                "response_id": rid,
                "source_id": sid,
                "question": src["question"],
                "context": src["context"],
                "original_answer": row["generated_response"],
                "split": (m or {}).get("split"),
                "y": (m or {}).get("y"),
                "gen_model": (m or {}).get("gen_model"),
                "gt_hallucination": row.get("hallucination"),
                "annotation_reason": row.get("annotation_reason"),
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"written: {n_ok} -> {args.out}", file=sys.stderr)
    if n_no_meta:
        print(f"not in metrics.csv (source_id derived from id): {n_no_meta}", file=sys.stderr)
    if n_no_src:
        print(f"skipped, no QA source_info: {n_no_src}", file=sys.stderr)


if __name__ == "__main__":
    main()
