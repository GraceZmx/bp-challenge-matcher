"""对照标准答案，评测"关键词匹配基线"和"完整系统"在合成评测集上的表现。

这是唯一同时读取标准答案和系统输出的程序；系统本身（src/）不读取 answers.json。
结果写进 docs/evaluation.md 里 <!-- AUTO:START --> 和 <!-- AUTO:END --> 之间的部分，其余人工撰写的内容保留。

用法（先用 BPM_DATASET=demo 跑完 src/ 的全流程）：
  .venv/bin/python eval/evaluate.py
  # 评测某次存档的结果，只打印不写文件：
  .venv/bin/python eval/evaluate.py --matches-dir demo_data/output/archive/v1_before_maturity_fix --no-write
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANSWERS_PATH = ROOT / "demo_data" / "answers.json"
MATCH_DIR = ROOT / "demo_data" / "output" / "matches"
DOC_PATH = ROOT / "docs" / "evaluation.md"
START, END = "<!-- AUTO:START -->", "<!-- AUTO:END -->"
K = 3
RECOMMEND_SCORE = 60   # 和 match_report.py 的"可以考虑"分档一致

ROLE_NAMES = {"gold_obvious": "明显答案", "gold_hidden": "陷阱b（关键词不同但能解决）",
              "decoy": "陷阱a（关键词像但解决不了）"}
SHORT_NAMES = {"gold_obvious": "明显答案", "gold_hidden": "陷阱b", "decoy": "陷阱a"}


def load_rankings(cid: str, match_dir: Path) -> tuple[dict, dict, dict, dict]:
    """返回三种排名 {bp: 名次} 和精排得分 {bp: 分数}。基线和粗筛覆盖全部 BP，精排只覆盖粗筛前 15"""
    coarse = json.loads((match_dir / "coarse" / f"{cid}.json").read_text(encoding="utf-8"))
    final = json.loads((match_dir / f"{cid}.json").read_text(encoding="utf-8"))
    baseline_rank = {r["bp"]: r["text_rank"] for r in coarse["all"]}
    coarse_rank = {r["bp"]: r["combined_rank"] for r in coarse["all"]}
    final_rank = {r["bp"]: r["final_rank"] for r in final["ranking"]}
    final_score = {r["bp"]: r["total"] for r in final["ranking"]}
    return baseline_rank, coarse_rank, final_rank, final_score


def fmt_rank(rank, in_top: bool, good: bool) -> str:
    if rank is None:
        return "未进入粗筛"
    mark = ("✅" if in_top else "❌") if good else ("❌" if in_top else "✅")
    return f"第 {rank} 名 {mark}"


def compute_metrics(match_dir: Path = MATCH_DIR) -> dict:
    """对照标准答案算指标；网页演示（app.py）也调用这个函数"""
    answers = json.loads(ANSWERS_PATH.read_text(encoding="utf-8"))["challenges"]
    totals = {m: {"hit": 0, "recall": 0.0, "top1": 0, "obvious": 0, "hidden": 0, "decoy_out": 0}
              for m in ("baseline", "system")}
    recommended = {"gold_obvious": 0, "gold_hidden": 0, "decoy": 0}  # 系统得分达到推荐线的个数
    per_challenge = {}

    for cid, ans in answers.items():
        *rank_list, score = load_rankings(cid, match_dir)
        ranks = dict(zip(("baseline", "coarse", "system"), rank_list))
        gold = {g["bp"]: g["type"] for g in ans["gold"]}
        decoy = ans["trap_keyword_decoy"]["bp"]
        role = {**gold, decoy: "decoy"}
        per_challenge[cid] = {"company": ans["company"], "ranks": ranks, "score": score, "role": role}

        for m in ("baseline", "system"):
            r = ranks[m]
            top = {bp for bp, n in r.items() if n <= K}
            hits = [bp for bp in gold if bp in top]
            t = totals[m]
            t["hit"] += bool(hits)
            t["recall"] += len(hits) / len(gold)
            t["top1"] += any(r.get(bp) == 1 for bp in gold)
            t["obvious"] += any(gold[bp] == "gold_obvious" for bp in hits)
            t["hidden"] += any(gold[bp] == "gold_hidden" for bp in hits)
            t["decoy_out"] += decoy not in top
        for bp, rl in role.items():
            recommended[rl] += score.get(bp, 0) >= RECOMMEND_SCORE

    return {"n": len(answers), "totals": totals, "recommended": recommended,
            "per_challenge": per_challenge}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches-dir", type=Path, default=MATCH_DIR, help="要评测的匹配结果目录")
    parser.add_argument("--no-write", action="store_true", help="只打印，不更新 docs/evaluation.md")
    args = parser.parse_args()

    metrics = compute_metrics(args.matches_dir)
    totals, recommended = metrics["totals"], metrics["recommended"]
    rows, trap_rows = [], []
    for cid, pc in metrics["per_challenge"].items():
        ranks, score, role = pc["ranks"], pc["score"], pc["role"]

        def top3(m):
            names = sorted((n, bp) for bp, n in ranks[m].items() if n <= K)
            return "<br>".join(f"{n}. {bp[:16]}{'（' + SHORT_NAMES[role[bp]] + '）' if bp in role else ''}"
                               for n, bp in names)
        rows.append(f"| {cid} | {pc['company']} | {top3('baseline')} | {top3('system')} |")

        for bp, rl in role.items():
            good = rl != "decoy"
            b, c, s = ranks["baseline"].get(bp), ranks["coarse"].get(bp), ranks["system"].get(bp)
            sc = score.get(bp)
            trap_rows.append(f"| {cid} | {SHORT_NAMES[rl]} | {bp[:22]} | {fmt_rank(b, b <= K, good)} | "
                             f"第 {c} 名 | {fmt_rank(s, s is not None and s <= K, good)} | "
                             f"{sc if sc is not None else '-'} |")

    n = metrics["n"]
    def pct(x):
        return f"{x}/{n}（{x / n:.0%}）" if isinstance(x, int) else f"{x / n:.0%}"
    b, s = totals["baseline"], totals["system"]
    lines = [START, "",
             "## 结果（由 eval/evaluate.py 自动生成）", "",
             "### 总体指标", "",
             "| 指标 | 关键词基线（TF-IDF） | 完整系统（粗筛+精排） |", "|---|---|---|",
             f"| Top-3 命中率（前 3 名里至少有 1 个正确答案） | {pct(b['hit'])} | {pct(s['hit'])} |",
             f"| Top-3 召回率（2 个正确答案平均找回几成） | {pct(b['recall'])} | {pct(s['recall'])} |",
             f"| Top-1 准确率（第 1 名就是正确答案） | {pct(b['top1'])} | {pct(s['top1'])} |",
             f"| 明显答案进入前 3 | {pct(b['obvious'])} | {pct(s['obvious'])} |",
             f"| 陷阱b 被识别（关键词不同但能解决的项目进入前 3） | {pct(b['hidden'])} | {pct(s['hidden'])} |",
             f"| 陷阱a 被识别（关键词像但解决不了的项目没进前 3） | {pct(b['decoy_out'])} | {pct(s['decoy_out'])} |",
             "", f"### 按推荐分数线看（完整系统得分 ≥ {RECOMMEND_SCORE} 分才算推荐）", "",
             "Top-3 只看相对名次：候选整体都弱时，得分很低的项目也会进前 3。这里再看系统是不是真的给了推荐分。", "",
             "| 角色 | 达到推荐线 |", "|---|---|",
             f"| 明显答案 | {pct(recommended['gold_obvious'])} |",
             f"| 陷阱b（应该推荐） | {pct(recommended['gold_hidden'])} |",
             f"| 陷阱a（不应该推荐） | {pct(recommended['decoy'])} |",
             "", "### 每道题的前 3 名", "",
             "| 题号 | 出题企业 | 关键词基线前 3 | 完整系统前 3 |", "|---|---|---|---|", *rows,
             "", "### 每个陷阱和正确答案的排名", "",
             "✅ 表示处理正确：正确答案进了前 3，或陷阱a 没进前 3。精排只评估粗筛前 15 名。", "",
             "| 题号 | 角色 | BP | 关键词基线 | 粗筛 | 完整系统 | 系统得分 |", "|---|---|---|---|---|---|---|",
             *trap_rows,
             "", END]
    auto = "\n".join(lines)

    if args.no_write:
        print(auto)
        return
    if DOC_PATH.exists() and START in DOC_PATH.read_text(encoding="utf-8"):
        doc = DOC_PATH.read_text(encoding="utf-8")
        doc = doc[:doc.index(START)] + auto + doc[doc.index(END) + len(END):]
    else:
        doc = "# 评测\n\n" + auto + "\n"
    DOC_PATH.write_text(doc, encoding="utf-8")
    print(auto)


if __name__ == "__main__":
    main()
