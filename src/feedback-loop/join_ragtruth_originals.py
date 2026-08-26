"""
Джойнит наш regen-context-by-response.jsonl с оригинальными текстами RAGTruth.

Скачайте локально (или клонируйте репозиторий):
  https://github.com/ParticleMedia/RAGTruth/blob/main/dataset/response.jsonl
  https://github.com/ParticleMedia/RAGTruth/blob/main/dataset/source_info.jsonl

Запуск:
  python3 join_ragtruth_originals.py \
      --response-jsonl path/to/response.jsonl \
      --source-info-jsonl path/to/source_info.jsonl \
      --regen-context path/to/regen-context-by-response.jsonl \
      --out ragtruth-originals-by-response.jsonl
"""
import json
import argparse
import hashlib


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_response_jsonl(path):
    """id -> {response, model, split, quality, labels}"""
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            idx[d["id"]] = d
    return idx


def load_source_info_jsonl(path):
    """source_id -> {question, passages, task_type}, только QA"""
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("task_type") != "QA":
                continue
            si = d["source_info"]
            idx[d["source_id"]] = {
                "question": si["question"],
                "passages": si["passages"],
            }
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response-jsonl", required=True)
    ap.add_argument("--source-info-jsonl", required=True)
    ap.add_argument("--regen-context", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extraction-summary", default=None,
                     help="опционально: support-critical/extraction_summary.json, "
                          "для сверки text_sha256")
    args = ap.parse_args()

    responses = load_response_jsonl(args.response_jsonl)
    sources = load_source_info_jsonl(args.source_info_jsonl)

    # опциональная сверка хэшей
    hash_check = {}
    if args.extraction_summary:
        gs = json.load(open(args.extraction_summary))["graph_records"]
        for rec in gs:
            hash_check[rec["response_id"]] = rec["cache"]["answer"]["text_sha256"]

    n_ok, n_missing_resp, n_missing_src, n_hash_mismatch = 0, 0, 0, 0
    with open(args.regen_context, encoding="utf-8") as fin, \
         open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            rid, sid = rec["response_id"], rec["source_id"]

            resp = responses.get(rid)
            src = sources.get(sid)
            if resp is None:
                n_missing_resp += 1
                continue
            if src is None:
                n_missing_src += 1
                continue

            expected_hash = hash_check.get(rid)
            if expected_hash is not None:
                actual_hash = sha256(resp["response"])
                if actual_hash != expected_hash:
                    n_hash_mismatch += 1
                    continue

            fout.write(json.dumps({
                "response_id": rid,
                "source_id": sid,
                "question": src["question"],
                "context": src["passages"],
                "original_answer": resp["response"],
                "gt_labels": resp.get("labels", []),   # эталонные спаны галлюцинаций из RAGTruth
                "quality": resp.get("quality"),
            }, ensure_ascii=False) + "\n")
            n_ok += 1

    print(f"joined: {n_ok}")
    print(f"missing in response.jsonl: {n_missing_resp}")
    print(f"missing in source_info.jsonl (не QA или нет source_id): {n_missing_src}")
    if hash_check:
        print(f"hash mismatches (пропущены): {n_hash_mismatch}")


if __name__ == "__main__":
    main()
