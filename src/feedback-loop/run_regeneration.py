"""
Runner for v1 fixed (full context + no-passage-artifacts instruction).
Same as run_regeneration.py but uses build_regen_prompt_v1_fixed.
"""
import json
import argparse
import time
import sys

from build_regen_prompt import build_prompt


def load_jsonl_index(path, key="response_id"):
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            idx[d[key]] = d
    return idx


def load_done_ids(out_path):
    done = set()
    try:
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    done.add(d["response_id"])
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        pass
    return done


def call_ollama(model, prompt, host=None, temperature=0.0, retries=2):
    import ollama
    client = ollama.Client(host=host) if host else ollama
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
            return resp["message"]["content"]
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    ap = argparse.ArgumentParser(description="Regeneration with v1 fixed (full context + no-passage-artifacts)")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--originals", default="ragtruth-originals-by-response.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="mistral")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--host", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-prompt", action="store_true")
    args = ap.parse_args()

    records = load_jsonl_index(args.in_path)

    try:
        originals = load_jsonl_index(args.originals)
    except FileNotFoundError:
        print(f"ERROR: {args.originals} not found.", file=sys.stderr)
        sys.exit(1)

    done_ids = load_done_ids(args.out) if args.resume else set()
    ids = [rid for rid in records if rid not in done_ids]
    if args.limit:
        ids = ids[:args.limit]

    print(f"input: {len(records)} | already done: {len(done_ids)} | to process: {len(ids)}",
          file=sys.stderr)

    mode = "a" if args.resume else "w"
    n_ok, n_skip, n_err = 0, 0, 0
    with open(args.out, mode, encoding="utf-8") as out:
        for i, rid in enumerate(ids, 1):
            record = records[rid]
            original = originals.get(rid)
            if original is None:
                n_skip += 1
                print(f"[{i}/{len(ids)}] {rid}: SKIP (no original)", file=sys.stderr)
                continue

            prompt = build_prompt(record, k=args.k, original=original)

            t0 = time.time()
            try:
                regenerated = call_ollama(
                    args.model, prompt, host=args.host, temperature=args.temperature,
                )
            except Exception as e:
                n_err += 1
                print(f"[{i}/{len(ids)}] {rid}: ERROR - {e}", file=sys.stderr)
                continue
            elapsed = time.time() - t0

            row = {
                "response_id": rid,
                "source_id": record["source_id"],
                "gen_model": record.get("gen_model"),
                "regen_model": args.model,
                "k": args.k,
                "prompt_version": "v1_fixed_full_context_no_artifacts",
                "question": original["question"],
                "context": original["context"],
                "original_answer": original["original_answer"],
                "regenerated_answer": regenerated,
                "elapsed_s": round(elapsed, 2),
            }
            if not args.no_prompt:
                row["prompt"] = prompt

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_ok += 1
            print(f"[{i}/{len(ids)}] {rid}: ok ({elapsed:.1f}s)", file=sys.stderr)

    print(f"done: {n_ok} ok, {n_skip} skipped, {n_err} errors -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()