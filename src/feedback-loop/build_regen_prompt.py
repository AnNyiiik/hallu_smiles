"""
Строит промпт на перегенерацию ответа по данным regen-context-by-response.jsonl.

Плейсхолдеры {question}/{context}/{original_answer} нужно заполнить самостоятельно —
сырых текстов в архиве нет (только офсеты и цитаты-выдержки). Сверить, что вы берёте
именно тот текст, можно по text_sha256 в support-critical/extraction_summary.json
(поле cache.answer/context/query.text_sha256) — это hash исходного текста.
"""
import json

def load_records(path="/home/claude/build/regen-context-by-response.jsonl"):
    idx = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            idx[d["response_id"]] = d
    return idx


def load_originals(path="ragtruth-originals-by-response.jsonl"):
    """Результат join_ragtruth_originals.py: response_id -> {question, context, original_answer}.
    Если файла нет (originals ещё не собраны), возвращает пустой индекс — тогда
    build_prompt оставит {question}/{context}/{original_answer} как плейсхолдеры."""
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
    """Ранжирование внутри unsupported/contradicted бинарное (risk=1 у обоих статусов),
    поэтому 'top-k' здесь = первые k по порядку появления в ответе (по start).
    Если k==3 и нужен ТОЧНО тот top-3, что в статье/пайплайне — берите его из
    top3-worst-claims-by-response.jsonl вместо этой функции."""
    claims = sorted(record["unsupported_or_contradicted_claims"], key=lambda c: c["start"])
    return claims[:k]

def format_evidence(ev_list):
    if not ev_list:
        return "      (референс не содержит близких по смыслу фрагментов вообще)"
    lines = []
    for e in ev_list:
        src = "контекст" if e["source"] == "context" else "запрос"
        lines.append(f"      - [{src}, пассаж #{e['index']}, офсет {e['start']}:{e['end']}] «{e['text']}»")
    return "\n".join(lines)

def build_prompt(record, k=3, original=None):
    """original: dict с {question, context, original_answer} из load_originals()[response_id],
    либо None — тогда в промпте останутся плейсхолдеры {question}/{context}/{original_answer}."""
    claims = top_k_claims(record, k=k)
    entities = record["ungrounded_entities"]

    claims_block = []
    for i, c in enumerate(claims, 1):
        verdict_ru = "противоречит источнику" if c["verdict"] == "contradicted" else "не подтверждено источником"
        claims_block.append(
            f"{i}. Утверждение [офсет в ответе {c['start']}:{c['end']}]: «{c['text']}»\n"
            f"   Статус: {verdict_ru}.\n"
            f"   Ближайшие фрагменты контекста/запроса (не подтверждают или опровергают это утверждение):\n"
            f"{format_evidence(c['reference_evidence'])}"
        )
    claims_block = "\n".join(claims_block) if claims_block else "(таких утверждений не найдено)"

    entities_block = []
    for i, e in enumerate(entities, 1):
        locs = e["locations_in_answer"]
        if locs:
            loc_str = "; ".join(f"[{l['start']}:{l['end']}] «{l['matched_text']}»" for l in locs)
            loc_str = f"встречается в ответе по офсету(ам): {loc_str}"
        else:
            loc_str = "дословно в тексте ответа не найдена (вероятно, извлечена в нормализованном/перефразированном виде) — точный офсет получить нельзя"
        rel = e["related_unsupported_relations"]
        rel_str = ("; ".join(f"({s} — {p} — {o})" for s, p, o in rel)) if rel else "нет"
        entities_block.append(
            f"{i}. Сущность «{e['entity']}»: {loc_str}.\n"
            f"   В контексте и запросе не найдено ни одного совпадения (ни по тексту, ни по смыслу) — "
            f"проверка релевантности для неё не проводилась вообще, поэтому конкретный опровергающий "
            f"фрагмент референса указать нельзя.\n"
            f"   Связанные с ней неподтверждённые утверждения графа ответа: {rel_str}"
        )
    entities_block = "\n".join(entities_block) if entities_block else "(таких сущностей не найдено)"

    prompt = f"""Ты редактируешь ответ модели на вопрос по данному контексту. Ниже — вопрос, контекст,
исходный ответ и список конкретных проблем в этом ответе, найденных при проверке на галлюцинации.
Перепиши ответ так, чтобы устранить именно эти проблемы, не меняя остальное и не добавляя новых
утверждений, которых нет в контексте или запросе.

## Вопрос
{{question}}

## Контекст (источник)
{{context}}

## Исходный ответ модели
{{original_answer}}

## Найденные проблемы

### Утверждения, не подтверждённые референсом (топ-{k})
{claims_block}

### Сущности без подтверждения в источниках
{entities_block}

## Задача
Перепиши ответ:
1. Убери или перефразируй утверждения из списка выше так, чтобы они либо опирались на приведённые
   фрагменты контекста/запроса, либо были удалены, если опоры нет вообще.
2. Убери или замени сущности из списка выше на те, что действительно есть в контексте/запросе,
   либо убери связанные с ними фразы целиком.
3. Не переформулируй и не трогай остальные части ответа, которые не упомянуты в списке проблем.
4. Не добавляй новых фактов, которых нет в контексте или запросе.
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
    rid = sys.argv[1] if len(sys.argv) > 1 else "12136"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(build_prompt(records[rid], k=k, original=originals.get(rid)))
