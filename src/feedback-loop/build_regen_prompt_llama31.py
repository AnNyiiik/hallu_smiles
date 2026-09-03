"""
Prompt builder for the llama-3.1 dataset (v1_fixed shape, adapted).

Differences from build_regen_prompt_v1_fixed.py, forced by what the archive stores:
  - no per-claim reference evidence -> the model is pointed at the Context
    section instead of at quoted fragments. The prompt never claims the
    reference "has nothing", because that isn't known.
  - no ungrounded entities -> that section is omitted entirely rather than
    rendered as "(no such entities found)", which would be false.
Everything else (full question + context, verdict-specific handling, the
no-passage-artifacts rule, output-only instruction) is unchanged.
"""
import json


def load_records(path="regen-context-llama31.jsonl"):
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            idx[d["response_id"]] = d
    return idx


def load_originals(path="ragtruth-originals-llama31.jsonl"):
    idx = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                idx[d["response_id"]] = d
    except FileNotFoundError:
        pass
    return idx


def top_k_claims(record, k=3):
    return sorted(record["unsupported_or_contradicted_claims"],
                  key=lambda c: c["start"])[:k]


def build_claim_instructions(claims):
    verdicts = {c["verdict"] for c in claims}
    lines = []
    if "contradicted" in verdicts:
        lines.append(
            '   - Claims marked "contradicts the source": the context states a '
            "different fact — find it in the Context section above and rewrite the "
            "claim to match it. Do not just delete it; correct it.")
    if "unsupported" in verdicts:
        lines.append(
            '   - Claims marked "not supported by the source": the context does not '
            "support this at all — remove the claim entirely rather than guessing a "
            "replacement.")
    return "\n".join(lines)


def build_prompt(record, k=3, original=None):
    claims = top_k_claims(record, k=k)
    if not claims:
        return None

    blocks = []
    for i, c in enumerate(claims, 1):
        verdict_en = ("contradicts the source" if c["verdict"] == "contradicted"
                      else "not supported by the source")
        blocks.append(
            f'{i}. Claim [answer offset {c["start"]}:{c["end"]}]: "{c["text"]}"\n'
            f"   Status: {verdict_en}.")
    claims_block = "\n".join(blocks)

    prompt = f"""You are editing a model's answer to a question, given a supporting context. Below are the
question, the context, the original answer, and a list of specific problems found in that answer
during hallucination checking. Rewrite the answer to fix exactly these problems, without changing
anything else and without adding any new claims that aren't in the context or the query.

## Question
{{question}}

## Context (source)
{{context}}

## Original model answer
{{original_answer}}

## Problems found

### Claims not supported by the reference (top-{k})
{claims_block}

## Task
Rewrite the answer:
1. Fix the claims listed above, handling each according to its status:
{build_claim_instructions(claims)}
2. Do not rephrase or touch any other part of the answer that isn't mentioned in the list of problems.
3. Do not add any new facts that aren't in the context or the query.
4. CRITICAL: Do NOT add references like "(passage 1:)", "(passage 2:)", "[passage 3]", or any other
   citations to passages. The context is provided for your reference only, not to be cited in the answer.
   Your rewritten answer should read naturally without these artifacts.

Output ONLY the rewritten answer text itself. Do not include any preamble ("Here is...", "Sure,..."),
explanation, notes about what you changed, or restatement of the problems list. Do not repeat these
instructions. Your entire response must be the corrected answer and nothing else.
"""
    if original:
        prompt = prompt.replace("{question}", original["question"])
        prompt = prompt.replace("{context}", original["context"])
        prompt = prompt.replace("{original_answer}", original["original_answer"])
    return prompt


if __name__ == "__main__":
    import sys
    records = load_records()
    originals = load_originals()
    rid = sys.argv[1] if len(sys.argv) > 1 else next(iter(records))
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(build_prompt(records[rid], k=k, original=originals.get(rid)))
