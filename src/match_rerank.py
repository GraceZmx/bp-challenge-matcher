"""匹配第二段：精排。让 DeepSeek 逐个评估粗筛出的 15 个候选，按评分标准打分。

评分（满分 100）：技术相关度 40、工艺环节匹配 25、成熟度 15、证据强度 20
只依据题目"推测"内容得到的分数，要列出来并扣分。总分由程序计算：各项得分 − 推测扣分。
模型引用的 BP 原文会和原文核对，找不到就重试。

每对 (题目, BP) 的结果单独存在 output/matches/rerank/<题号>/<BP>.json，已有的会跳过；
汇总到 output/matches/<题号>.json（含前 3 名）。
用法：.venv/bin/python src/match_rerank.py
"""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from extract_structured import MODEL, NA, load_client, squash
from match_coarse import BP_STRUCT_DIR, BP_TEXT_DIR, CHALLENGE_DIR, OUT_DIR as COARSE_DIR
from paths import OUTPUT_DIR

MATCH_DIR = OUTPUT_DIR / "matches"
RERANK_DIR = MATCH_DIR / "rerank"
ERROR_LOG = MATCH_DIR / "rerank_errors.log"
TOP_N = 3

CRITERIA = {  # 字段名: (中文名, 满分)
    "tech_relevance": ("技术相关度", 40),
    "process_match": ("工艺环节匹配", 25),
    "maturity": ("成熟度", 15),
    "evidence_strength": ("证据强度", 20),
}

SYSTEM_PROMPT = """你是生物制造领域的技术转移专家，负责评估一个参赛项目（BP）能否解决某家企业提出的技术难题。

【评分标准】满分 100：
1. tech_relevance 技术相关度（0-40）：BP 的核心技术和产品能否直接解决题目的核心问题
   - 33-40：同一产物或同一技术问题，可直接对口
   - 21-32：技术路线相同或相近，产物不同但能较快迁移
   - 9-20：只有部分环节或通用能力相关
   - 0-8：基本无关
2. process_match 工艺环节匹配（0-25）：题目涉及的工艺环节，BP 原文是否显示有实际能力（不是只提到名词）
3. maturity 成熟度（0-15）：题目有最低成熟度要求时，看 BP 是否达到；题目没有要求时，按 BP 当前阶段给分：
   量产 15、中试 12、小试 8、实验室 4、无法判断 6
   成熟度只看 BP 自身的阶段，不参与下面的推测扣分
4. evidence_strength 证据强度（0-20）：BP 对上述相关能力有没有量化数据、对照条件和可核查的证据；
   原文自己标注"无可核对数值""待核实"的，要相应低分

【推测扣分】
题目里每条内容都标了［原文］或［推测］。如果某一项的得分，只是因为 BP 和题目的［推测］内容吻合，
而题目［原文］内容并不支持，就要把这部分得分扣掉一半，写进 inference_deductions。
- 要扣：题目"发酵/培养［推测］"，BP 确实有发酵能力，你因此在工艺环节匹配里给了 6 分 → 扣 3 分并记录
- 不要扣：题目"基因编辑［推测］"，BP 没有提到基因编辑 → BP 没对上，本来就没因此得分，不属于推测扣分，不要记录
  （BP 缺少某种能力，应该体现在该项的低分和 gaps_risks 里，而不是 inference_deductions）
没有需要扣的就填空列表。

【其他要求】
- 只依据给出的材料，不要编造 BP 没有写的能力或数据
- evidence_quotes：1-4 条支持你判断的 BP 原文片段，必须一字不差地照抄 BP 原文
- 给分要拉开差距，不相关的候选就给低分

【输出格式】只输出一个 JSON 对象：
{
  "tech_relevance": {"score": 整数, "comment": "一句话理由"},
  "process_match": {"score": 整数, "comment": "..."},
  "maturity": {"score": 整数, "comment": "..."},
  "evidence_strength": {"score": 整数, "comment": "..."},
  "inference_deductions": [{"criterion": "上面四项之一的字段名", "points": 扣掉的分数(正整数), "reason": "依据的是哪条推测内容"}],
  "recommendation": "推荐理由，2-3 句",
  "evidence_quotes": ["BP 原文片段", ...],
  "gaps_risks": ["存在的差距或风险", ...]
}"""


def tagged(item) -> str:
    return f"{item['value']}［{item['source']}］" if item else "无"


def describe_challenge(ch: dict) -> str:
    lines = [f"出题企业：{ch['company']}",
             f"核心问题：{tagged(ch['core_problem'])}",
             f"补充理解：{tagged(ch['core_problem_inferred'])}"]
    for field, label in [("process_stages", "工艺环节"), ("application_categories", "应用领域"),
                         ("target_products", "目标产物"), ("tech_keywords", "技术关键词")]:
        lines.append(f"{label}：{'、'.join(tagged(x) for x in ch[field]) or '无'}")
    # 只有原文写明的成熟度要求才拿来要求 BP；推测出来的要求不给模型看，避免先按推测要求评分、再按推测扣分
    mm = ch["min_maturity"]
    lines.append(f"最低成熟度要求：{mm['value'] if mm['source'] == '原文' and mm['value'] != NA else '无（原文没有写）'}")
    return "\n".join(lines)


def describe_bp(bp: dict, text: str) -> str:
    return (f"【结构化信息】\n工艺环节：{'、'.join(bp['process_stages'])}\n"
            f"应用领域：{'、'.join(bp['application_categories'])}\n"
            f"产业化阶段：{bp['maturity']}（依据：{bp['maturity_evidence']}）\n\n"
            f"【BP 原文】\n{text}")


def validate(data: dict, bp_text: str) -> dict:
    for key, (name, full) in CRITERIA.items():
        s = data.get(key, {}).get("score")
        if not isinstance(s, int) or not 0 <= s <= full:
            raise ValueError(f"{key}（{name}）得分应为 0-{full} 的整数：{s}")
    deductions = data.get("inference_deductions")
    if not isinstance(deductions, list):
        raise ValueError("inference_deductions 应为列表")
    for d in deductions:
        if d.get("criterion") not in CRITERIA or not isinstance(d.get("points"), int) or d["points"] <= 0:
            raise ValueError(f"inference_deductions 条目格式不对：{d}")
        if d["criterion"] == "maturity":
            raise ValueError("成熟度只看 BP 自身阶段，不参与推测扣分")
        if d["points"] > data[d["criterion"]]["score"]:
            raise ValueError(f"{d['criterion']} 扣分不能超过该项得分：{d}")
    quotes = data.get("evidence_quotes")
    if not isinstance(quotes, list) or not quotes:
        raise ValueError("evidence_quotes 至少要有 1 条")
    missing = [q for q in quotes if squash(q) not in squash(bp_text)]
    if missing:
        raise ValueError(f"evidence_quotes 在 BP 原文中找不到：{missing}")
    for key in ("recommendation", "gaps_risks"):
        if key not in data:
            raise ValueError(f"缺少字段：{key}")

    raw = sum(data[k]["score"] for k in CRITERIA)
    deducted = sum(d["points"] for d in deductions)
    data["raw_total"] = raw
    data["inference_deducted"] = deducted
    data["total"] = raw - deducted
    data["relies_on_inference"] = bool(deductions)
    return data


def score_one(client, ch: dict, bp: dict, bp_text: str, last_err: str | None) -> dict:
    prompt = (f"=== 企业难题 ===\n{describe_challenge(ch)}\n\n"
              f"=== 候选 BP ===\n{describe_bp(bp, bp_text)}")
    if last_err:
        prompt += f"\n\n注意：你上一次的输出没有通过检查，原因是：{last_err}\n请修正后重新输出完整的 JSON。"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return validate(json.loads(resp.choices[0].message.content), bp_text)


def process(client, ch: dict, cand: dict) -> tuple[str, str, str | None]:
    out = RERANK_DIR / ch["id"] / f"{cand['bp']}.json"
    bp = json.loads((BP_STRUCT_DIR / f"{cand['bp']}.json").read_text(encoding="utf-8"))
    bp_text = (BP_TEXT_DIR / f"{cand['bp']}.txt").read_text(encoding="utf-8")
    last_err = None
    for _ in range(2):  # 第一次 + 重试一次（重试时告诉模型上次错在哪）
        try:
            result = score_one(client, ch, bp, bp_text, last_err)
            result.update({"bp": cand["bp"], "project_name": bp["project_name"],
                           "company_name": bp["company_name"], "coarse": cand})
            out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return ch["id"], cand["bp"], None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return ch["id"], cand["bp"], last_err


def summarize(ch: dict, candidates: list[dict]):
    results = []
    for cand in candidates:
        f = RERANK_DIR / ch["id"] / f"{cand['bp']}.json"
        if f.exists():
            results.append(json.loads(f.read_text(encoding="utf-8")))
    results.sort(key=lambda r: (-r["total"], r["coarse"]["combined_rank"]))
    for rank, r in enumerate(results, 1):
        r["final_rank"] = rank
    summary = {"challenge_id": ch["id"], "company": ch["company"], "title": ch["title"],
               "scored": len(results), "candidates": len(candidates),
               "top": results[:TOP_N], "ranking": results}
    (MATCH_DIR / f"{ch['id']}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def main():
    challenges = [json.loads(f.read_text(encoding="utf-8"))
                  for f in sorted(CHALLENGE_DIR.glob("*.json"))]
    jobs = []
    for ch in challenges:
        (RERANK_DIR / ch["id"]).mkdir(parents=True, exist_ok=True)
        coarse = json.loads((COARSE_DIR / f"{ch['id']}.json").read_text(encoding="utf-8"))
        ch["_candidates"] = coarse["candidates"]
        jobs += [(ch, c) for c in coarse["candidates"]
                 if not (RERANK_DIR / ch["id"] / f"{c['bp']}.json").exists()]
    print(f"共 {sum(len(c['_candidates']) for c in challenges)} 对需要评分，本次评分 {len(jobs)} 对")

    client = load_client()
    errors = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(process, client, ch, c) for ch, c in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            cid, bp, err = fut.result()
            if err:
                errors.append((cid, bp, err))
                print(f"[{i}/{len(jobs)}] ❌ {cid} {bp[:30]}：{err[:80]}")
            elif i % 15 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] 进行中…")

    if errors:
        with ERROR_LOG.open("a", encoding="utf-8") as fp:
            for cid, bp, err in errors:
                fp.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{cid}\t{bp}\t{err}\n")

    print(f"\n本次成功 {len(jobs) - len(errors)} 对，失败 {len(errors)} 对")
    for ch in challenges:
        results = summarize(ch, ch["_candidates"])
        print(f"\n=== {ch['id']} {ch['company']}（已评分 {len(results)}/{len(ch['_candidates'])}）")
        for r in results[:TOP_N]:
            flag = " ⚠️推测扣分" if r["relies_on_inference"] else ""
            print(f"  {r['final_rank']}. {r['total']}分（粗筛第{r['coarse']['combined_rank']}）"
                  f" {r['project_name'][:30]}{flag}")


if __name__ == "__main__":
    main()
