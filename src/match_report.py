"""根据 output/matches/ 的结果生成中文报告 output/match_report.md（不调用大模型）。

用法：.venv/bin/python src/match_report.py
"""
import json
from datetime import datetime

from match_coarse import BP_STRUCT_DIR, CHALLENGE_DIR, TAG_WEIGHT, TEXT_WEIGHT, TOP_K
from match_rerank import CRITERIA, MATCH_DIR, TOP_N
from paths import DATASET, OUTPUT_DIR, ROOT

REPORT_PATH = OUTPUT_DIR / "match_report.md"
BASELINE_PATH = MATCH_DIR / "baseline_tfidf.json"
# 参考分档，只用于阅读，不影响排名
LEVELS = [(75, "推荐对接"), (60, "可以考虑"), (0, "匹配度低")]


def level(score: int) -> str:
    return next(name for threshold, name in LEVELS if score >= threshold)


def tagged(item) -> str:
    if not item:
        return "无"
    return item["value"] + ("（推测）" if item["source"] == "推测" else "")


def challenge_section(ch: dict, summary: dict, coarse: dict, baseline: list) -> list[str]:
    lines = [f"## {ch['id']} {ch['company']}", ""]
    lines += [f"- **核心问题（原文）**：{ch['core_problem']['value']}",
              f"- **补充理解（推测）**：{tagged(ch['core_problem_inferred'])}",
              f"- **工艺环节**：{'、'.join(tagged(x) for x in ch['process_stages']) or '无'}",
              f"- **应用领域**：{'、'.join(tagged(x) for x in ch['application_categories']) or '无'}",
              f"- **目标产物**：{'、'.join(tagged(x) for x in ch['target_products']) or '无'}"]
    if ch["removed_content"]:
        lines.append(f"- **已删除的非技术内容**：{'；'.join(ch['removed_content'])}")
    if ch.get("manual_overrides"):
        lines.append(f"- **人工修正**：{ch['manual_overrides']['note']}")
    if coarse["excluded_self"]:
        lines.append(f"- **已排除出题企业自己的 BP**：{'；'.join(coarse['excluded_self'])}")
    lines.append("")

    header = "| 排名 | 项目 | 团队/公司 | 总分 | " + " | ".join(
        f"{name}({full})" for name, full in CRITERIA.values()) + " | 推测扣分 | 结论 |"
    lines += [f"### 前 {TOP_N} 名", "", header, "|" + "---|" * (len(CRITERIA) + 6)]
    for r in summary["top"]:
        scores = " | ".join(str(r[k]["score"]) for k in CRITERIA)
        lines.append(f"| {r['final_rank']} | {r['project_name']} | {r['company_name']} | **{r['total']}** | "
                     f"{scores} | {('-' + str(r['inference_deducted'])) if r['inference_deducted'] else '0'} | "
                     f"{level(r['total'])} |")
    lines.append("")

    for r in summary["top"]:
        lines += [f"#### 第 {r['final_rank']} 名：{r['project_name']}（{r['total']} 分）", "",
                  f"**推荐理由**：{r['recommendation']}", "", "**BP 原文证据**："]
        lines += [f"> - {q}" for q in r["evidence_quotes"]]
        lines += ["", "**差距或风险**："] + [f"- {g}" for g in r["gaps_risks"]]
        if r["inference_deductions"]:
            lines += ["", "**⚠️ 依据推测内容的部分（已扣分）**："]
            lines += [f"- {CRITERIA[d['criterion']][0]} −{d['points']}：{d['reason']}"
                      for d in r["inference_deductions"]]
        lines.append("")

    # 和关键词基线对比
    base_rank = {b["bp"]: b["text_rank"] for b in baseline}
    lines += ["### 和关键词匹配基线对比", "",
              f"基线只按 TF-IDF 文本相似度排名。粗筛前 {TOP_K} 名与基线前 {TOP_K} 名重合 "
              f"{len({c['bp'] for c in coarse['candidates']} & set(base_rank))} 个。", "",
              "| 最终排名 | 项目 | 在基线中的排名 |", "|---|---|---|"]
    for r in summary["top"]:
        lines.append(f"| {r['final_rank']} | {r['project_name']} | 第 {r['coarse']['text_rank']} 名 |")
    lines += ["", f"基线前 {TOP_N} 名：" + "；".join(
        f"{b['text_rank']}. {b['project_name']}" for b in baseline[:TOP_N]), ""]
    return lines


def main():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    challenges = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(CHALLENGE_DIR.glob("*.json"))]
    data = []
    for ch in challenges:
        summary = json.loads((MATCH_DIR / f"{ch['id']}.json").read_text(encoding="utf-8"))
        coarse = json.loads((MATCH_DIR / "coarse" / f"{ch['id']}.json").read_text(encoding="utf-8"))
        data.append((ch, summary, coarse, baseline[ch["id"]]))

    n_bp = len(list(BP_STRUCT_DIR.glob("*.json")))
    title = "企业难题 × BP 智能匹配报告" + ("（合成数据 synthetic）" if DATASET == "demo" else "")
    lines = [f"# {title}", "",
             f"生成时间：{datetime.now():%Y-%m-%d %H:%M}　|　企业难题 {len(challenges)} 道　|　BP {n_bp} 份", "",
             "## 方法", "",
             f"1. **粗筛（不调用大模型）**：每道题从 {n_bp} 份 BP 里挑 {TOP_K} 个候选。",
             f"   - 标签重合度（权重 {TAG_WEIGHT}）：工艺环节、应用领域重合越多分越高，题目里标\"推测\"的标签只算一半",
             f"   - 文本相似度（权重 {TEXT_WEIGHT}）：TF-IDF，按字切分（单字 + 相邻两字），题目里推测的内容只算一半权重",
             "   - 出题企业自己的 BP 不参与匹配",
             f"2. **精排（DeepSeek）**：逐个评估 {TOP_K} 个候选，满分 100："
             + "、".join(f"{n} {f}" for n, f in CRITERIA.values()) + "。",
             "   - 只依据题目\"推测\"内容得到的分数扣掉一半，并单独列出",
             "   - 引用的 BP 原文由程序核对，总分由程序计算",
             f"3. **结论分档（仅供参考）**：" + "、".join(
                 f"{t} 分以上{n}" if t else f"{LEVELS[i - 1][0]} 分以下{n}" for i, (t, n) in enumerate(LEVELS)),
             "", "## 总览", "",
             "| 题号 | 出题企业 | 第 1 名 | 第 2 名 | 第 3 名 |", "|---|---|---|---|---|"]
    for ch, summary, _, _ in data:
        cells = [f"{r['project_name'][:20]}（{r['total']}，{level(r['total'])}）" for r in summary["top"]]
        lines.append(f"| {ch['id']} | {ch['company']} | " + " | ".join(cells) + " |")
    lines.append("")
    for ch, summary, coarse, base in data:
        lines += challenge_section(ch, summary, coarse, base)

    if DATASET == "real":
        lines += REAL_DATA_LIMITATIONS
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"报告已生成：{REPORT_PATH.relative_to(ROOT)}（{len(lines)} 行）")


# 真实数据人工检查后发现的局限（合成数据的评测结果见 docs/evaluation.md）
REAL_DATA_LIMITATIONS = ["## 已知局限", "",
              "- 企业难题原文信息很少（大多只有一句标题），应用领域全部、工艺环节约一半是推测，匹配结果的可信度受此限制",
              "- 标签太宽泛时（例如只有\"发酵/培养\"\"菌种/细胞构建\"），很多 BP 标签分都是满分，粗筛会放进文本相似度很低的候选",
              "- 推测扣分由模型判断，抽查发现约 7% 的扣分条目理由不成立（例如把原文里的\"高密度发酵\"当成推测），结果仍需人工复核",
              "  - 例：C04 第 2 名被扣 3 分，理由是\"发酵能力吻合推测内容\"，但 C04 题目并没有发酵这个标签，"
              "这 3 分不该扣，实际应为 70 分，会高于现在的第 1 名（69 分）。报告保留程序原始结果，不手动改分，"
              "C04 的前两名应视为不分先后",
              "- 精排打分有随机波动：同样的输入重跑一次，每对总分平均变化约 3 分、最大约 12 分，"
              "前 3 名可能换掉一个项目。相差 5 分以内的候选不应当作有明确先后",
              "- BP 本身是一页纸的项目简介，很多指标标注\"待核实\"，证据强度普遍偏低", ""]


if __name__ == "__main__":
    main()
