"""
Builds regen-context for the llama-3.1 controlled dataset.

This archive has no audit/ records, so claims and verdicts are reconstructed:
  - claims: critical_claims + critical_coverage cache files, matched to responses
    by content (a claim's [start:end] must slice exactly out of the answer),
    then deduplicated by (start, end, text) — a rule that reproduces the
    per-response claim count in 681 of 684 matched responses.
  - verdicts: the ordered critical_claim_statuses column in metrics.csv.

The claim->verdict link is positional, and two plausible orderings exist
(first-appearance vs sorted-by-offset). For 146 responses those orderings
disagree about WHICH claims are flagged, so this script emits only responses
where the flagged set is identical under both orderings. Everything it writes
is therefore ordering-independent; ambiguous responses are dropped, not guessed.

Not available in this archive (and so absent from the output):
  - reference evidence per claim (critical_verdicts stores only {"verdict": ...})
  - ungrounded entities
"""
import json, glob, csv, argparse, sys, os

FLAG = {"unsupported", "contradicted"}


def matches(answer, claims):
    return bool(claims) and all(
        answer[c["start"]:c["end"]] == c["text"] for c in claims)


def index_cache(root, subdir, originals):
    idx = {}
    for cf in glob.glob(os.path.join(root, "data/cache", subdir, "*.json")):
        claims = json.load(open(cf))["claims"]
        hits = [rid for rid, o in originals.items()
                if matches(o["original_answer"], claims)]
        if len(hits) == 1:
            idx.setdefault(hits[0], []).extend(claims)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive-root", required=True)
    ap.add_argument("--originals", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    originals = {}
    with open(args.originals, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            originals[d["response_id"]] = d

    atomic = index_cache(args.archive_root, "critical_claims", originals)
    coverage = index_cache(args.archive_root, "critical_coverage", originals)

    metrics = glob.glob(os.path.join(
        args.archive_root, "data/runs/*/support-critical/metrics.csv"))[0]

    key = lambda c: (c["start"], c["end"], c["text"])
    n_out = n_count_mismatch = n_ambiguous = n_no_claims = 0

    with open(args.out, "w", encoding="utf-8") as out:
        for row in csv.DictReader(open(metrics, encoding="utf-8")):
            rid = row["response_id"]
            if rid not in atomic:
                n_no_claims += 1
                continue
            statuses = json.loads(row["critical_claim_statuses"])

            seen, order = set(), []
            for c in atomic.get(rid, []) + coverage.get(rid, []):
                if key(c) not in seen:
                    seen.add(key(c))
                    order.append(c)

            if len(order) != len(statuses):
                n_count_mismatch += 1
                continue

            srt = sorted(order, key=key)
            flagged_a = {key(c) for c, s in zip(order, statuses) if s in FLAG}
            flagged_b = {key(c) for c, s in zip(srt, statuses) if s in FLAG}
            if flagged_a != flagged_b:
                n_ambiguous += 1
                continue

            verdict_of = {key(c): s for c, s in zip(srt, statuses)}
            claims_out = [{
                "text": c["text"],
                "start": c["start"],
                "end": c["end"],
                "verdict": verdict_of[key(c)],
                "reference_evidence": [],
            } for c in srt if verdict_of[key(c)] in FLAG]

            if not claims_out:
                continue

            out.write(json.dumps({
                "response_id": rid,
                "source_id": row["source_id"],
                "gen_model": row["gen_model"],
                "split": row["split"],
                "gt_label": row["y"],
                "H_support_critical": row.get("H_support_critical"),
                "EG": row.get("EG"),
                "claim_count_total": len(order),
                "evidence_available": False,
                "entities_available": False,
                "ungrounded_entities": [],
                "unsupported_or_contradicted_claims": claims_out,
            }, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"written: {n_out} -> {args.out}", file=sys.stderr)
    print(f"dropped — claim cache not matched:    {n_no_claims}", file=sys.stderr)
    print(f"dropped — claim/verdict count differs: {n_count_mismatch}", file=sys.stderr)
    print(f"dropped — flagged set order-ambiguous: {n_ambiguous}", file=sys.stderr)


if __name__ == "__main__":
    main()
