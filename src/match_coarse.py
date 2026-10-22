"""匹配第一段：粗筛（不调用大模型）。

给每道企业难题，从全部 BP 里挑 15 个候选：
  (a) 标签重合度：工艺环节、应用领域重合越多分越高；题目里标"推测"的标签只算一半
  (b) 文本相似度：TF-IDF（按字切分：单字 + 相邻两字），比较题目和 BP 原文
  综合分 = TAG_WEIGHT × (a) + TEXT_WEIGHT × (b)，取前 15
另外只按 (b) 排名的前 15 单独存为"关键词匹配基线"，以后用来对比。

结果：output/matches/coarse/<题号>.json，output/matches/baseline_tfidf.json
用法：.venv/bin/python src/match_coarse.py
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from paths import OUTPUT_DIR, ROOT, TEXT_DIR

BP_STRUCT_DIR = OUTPUT_DIR / "structured"
BP_TEXT_DIR = TEXT_DIR
CHALLENGE_DIR = OUTPUT_DIR / "challenges_structured"
OUT_DIR = OUTPUT_DIR / "matches" / "coarse"
BASELINE_PATH = OUTPUT_DIR / "matches" / "baseline_tfidf.json"

TOP_K = 15
TAG_WEIGHT = 0.5
TEXT_WEIGHT = 0.5
SOURCE_WEIGHT = {"原文": 1.0, "推测": 0.5}
TAG_FIELDS = ["process_stages", "application_categories"]


def load_bps() -> list[dict]:
    bps = []
    for f in sorted(BP_STRUCT_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        data["text"] = (BP_TEXT_DIR / f"{f.stem}.txt").read_text(encoding="utf-8")
        bps.append(data)
    return bps


def load_challenges() -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(CHALLENGE_DIR.glob("*.json"))]


def tag_score(ch: dict, bp: dict) -> tuple[float, list[str]]:
    """返回 (0~1 的分数, 对上的标签)"""
    total, hit, matched = 0.0, 0.0, []
    for field in TAG_FIELDS:
        bp_tags = set(bp[field])
        for item in ch[field]:
            w = SOURCE_WEIGHT[item["source"]]
            total += w
            if item["value"] in bp_tags:
                hit += w
                matched.append(f"{item['value']}［{item['source']}］")
    return (hit / total if total else 0.0), matched


def query_texts(ch: dict) -> tuple[str, str]:
    """题目里用于文本比对的内容，分成 (原文部分, 推测部分)"""
    items = [ch["core_problem"], ch["core_problem_inferred"],
             *ch["target_products"], *ch["tech_keywords"]]
    orig = [x["value"] for x in items if x and x["source"] == "原文"]
    inferred = [x["value"] for x in items if x and x["source"] == "推测"]
    return " ".join(orig), " ".join(inferred)


def squash(s: str) -> str:
    return "".join(s.split())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bps, challenges = load_bps(), load_challenges()

    # IDF 只用 BP 来算；sublinear_tf 防止某个字重复很多次就把分数拉得很高
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), sublinear_tf=True,
                                 preprocessor=squash)
    bp_matrix = vectorizer.fit_transform([bp["text"] for bp in bps])

    baseline = {}
    for ch in challenges:
        orig, inferred = query_texts(ch)
        # 推测部分只算一半权重
        q = vectorizer.transform([orig]) + SOURCE_WEIGHT["推测"] * vectorizer.transform([inferred])
        text_scores = (bp_matrix @ normalize(q).T).toarray().ravel()
        # 出题企业自己的 BP 不能作为解决方案推荐给它自己
        is_self = np.array([squash(bp["company_name"]) == squash(ch["company"]) for bp in bps])
        excluded = [bp["project_name"] for bp, s in zip(bps, is_self) if s]
        text_scores = np.where(is_self, 0.0, text_scores)
        text_norm = text_scores / text_scores.max() if text_scores.max() > 0 else text_scores

        rows = []
        for bp, ts, tn, s in zip(bps, text_scores, text_norm, is_self):
            if s:
                continue
            tg, matched = tag_score(ch, bp)
            rows.append({
                "bp": bp["source_file"],
                "project_name": bp["project_name"],
                "tag_score": round(tg, 4),
                "matched_tags": matched,
                "text_score": round(float(ts), 4),
                "text_score_norm": round(float(tn), 4),
                "combined": round(TAG_WEIGHT * tg + TEXT_WEIGHT * float(tn), 4),
            })

        by_text = sorted(rows, key=lambda r: -r["text_score"])
        for rank, r in enumerate(by_text, 1):
            r["text_rank"] = rank
        by_combined = sorted(rows, key=lambda r: (-r["combined"], -r["text_score"]))
        for rank, r in enumerate(by_combined, 1):
            r["combined_rank"] = rank

        baseline[ch["id"]] = [{k: r[k] for k in ("text_rank", "bp", "project_name", "text_score")}
                              for r in by_text[:TOP_K]]
        out = {"challenge_id": ch["id"], "company": ch["company"],
               "query_original": orig, "query_inferred": inferred,
               "weights": {"tag": TAG_WEIGHT, "text": TEXT_WEIGHT},
               "excluded_self": excluded,
               "candidates": by_combined[:TOP_K], "all": by_combined}
        (OUT_DIR / f"{ch['id']}.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

        overlap = len({r["bp"] for r in by_combined[:TOP_K]} & {r["bp"] for r in by_text[:TOP_K]})
        print(f"\n=== {ch['id']} {ch['company']}（与基线前{TOP_K}重合 {overlap} 个）")
        if excluded:
            print(f"  （已排除出题企业自己的 BP：{excluded}）")
        for r in by_combined[:TOP_K]:
            print(f"  {r['combined_rank']:>2}. 综合{r['combined']:.2f} 标签{r['tag_score']:.2f} "
                  f"文本{r['text_score_norm']:.2f}(第{r['text_rank']}) {r['project_name'][:28]}")

    BASELINE_PATH.write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n粗筛结果已存到 {OUT_DIR.relative_to(ROOT)}/，基线存到 {BASELINE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
