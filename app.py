"""企业难题 × BP 智能匹配：网页演示（Streamlit）。

只读取 demo_data/ 里的合成数据和已经跑好的结果，不调用大模型，也不读取 data/、output/ 里的真实数据。

启动：.venv/bin/streamlit run app.py
"""
import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo_data"                      # 只读这个目录
CHALLENGE_DIR = DEMO / "output" / "challenges_structured"
MATCH_DIR = DEMO / "output" / "matches"
BP_DIR = DEMO / "bps"
ANSWERS_PATH = DEMO / "answers.json"

sys.path.insert(0, str(ROOT / "eval"))
from evaluate import K, RECOMMEND_SCORE, compute_metrics  # noqa: E402  和 docs/evaluation.md 用同一份算法

CRITERIA = {"tech_relevance": ("技术相关度", 40), "process_match": ("工艺环节匹配", 25),
            "maturity": ("成熟度", 15), "evidence_strength": ("证据强度", 20)}
LONG_TAG_CHARS = 14   # 超过这个字数的标签改用圆角矩形框
ROLE_BADGE = {"gold_obvious": ("正确答案", "ok"), "gold_hidden": ("正确答案 · 陷阱b", "ok"),
              "decoy": ("陷阱a", "bad")}

st.set_page_config(page_title="企业难题 × BP 智能匹配", page_icon="🧬", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.flow {display:flex; flex-wrap:wrap; align-items:stretch; gap:.4rem; margin:.6rem 0 1.2rem;}
.flow .step {flex:1 1 150px; border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:.6rem .8rem;
             background:rgba(99,102,241,.06);}
.flow .step b {display:block; font-size:1rem;}
.flow .step span {font-size:.8rem; opacity:.75;}
.flow .step.llm {background:rgba(16,185,129,.08);}
.flow .arrow {display:flex; align-items:center; opacity:.45; font-size:1.2rem;}
.tag {display:inline-block; padding:.12rem .55rem; margin:.12rem .25rem .12rem 0; border-radius:999px;
      font-size:.85rem; line-height:1.5;}
.tag.long {display:block; border-radius:8px; padding:.4rem .7rem; margin:.2rem 0; line-height:1.65;}
.tag.orig {background:rgba(16,185,129,.14); color:#0f8a5f; border:1px solid rgba(16,185,129,.45);}
.tag.inf {background:rgba(245,158,11,.12); color:#b45309; border:1px dashed rgba(245,158,11,.7);}
.tag.ok {background:rgba(16,185,129,.14); color:#0f8a5f;}
.tag.bad {background:rgba(239,68,68,.12); color:#c2410c;}
.tag.none {background:rgba(128,128,128,.12); opacity:.8;}
.row {display:flex; gap:.8rem; padding:.35rem 0; border-bottom:1px solid rgba(128,128,128,.12);}
.row .k {flex:0 0 7rem; opacity:.65; font-size:.9rem; padding-top:.15rem;}
.removed {text-decoration:line-through; opacity:.55; font-size:.88rem;}
.score {font-size:2.4rem; font-weight:700; line-height:1;}
.score small {font-size:.9rem; font-weight:400; opacity:.6;}
.bar {height:8px; border-radius:4px; background:rgba(128,128,128,.18); overflow:hidden; margin:.15rem 0 .45rem;}
.bar > div {height:100%; border-radius:4px; background:#6366f1;}
.bar-label {display:flex; justify-content:space-between; font-size:.82rem; opacity:.85;}
.quote {border-left:3px solid rgba(99,102,241,.5); padding:.25rem .7rem; margin:.3rem 0; font-size:.88rem;
        background:rgba(99,102,241,.05); border-radius:0 6px 6px 0;}
.rank-item {padding:.55rem .7rem; border-radius:8px; margin-bottom:.45rem; border:1px solid rgba(128,128,128,.2);}
.rank-item.ok {border-color:rgba(16,185,129,.55); background:rgba(16,185,129,.06);}
.rank-item.bad {border-color:rgba(239,68,68,.55); background:rgba(239,68,68,.06);}
.rank-item .meta {font-size:.8rem; opacity:.65;}
</style>
""", unsafe_allow_html=True)


# ---------- 读数据（全部来自 demo_data/） ----------
@st.cache_data
def load_json(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data
def load_challenges() -> dict:
    return {f.stem: load_json(str(f)) for f in sorted(CHALLENGE_DIR.glob("*.json"))}


@st.cache_data
def load_metrics() -> dict:
    return compute_metrics(MATCH_DIR)


def esc(s) -> str:
    return html.escape(str(s))


def tags(items, empty="无") -> str:
    if isinstance(items, dict):
        items = [items]
    items = [x for x in items if x]
    if not items:
        return f'<span class="tag none">{empty}</span>'
    return "".join(f'<span class="tag {"orig" if x["source"] == "原文" else "inf"}'
                   f'{" long" if len(str(x["value"])) > LONG_TAG_CHARS else ""}" '
                   f'title="{esc(x["basis"])}">{esc(x["value"])}</span>' for x in items)


def level(score: int) -> tuple[str, str]:
    if score >= 75:
        return "推荐对接", "#0f8a5f"
    if score >= RECOMMEND_SCORE:
        return "可以考虑", "#b45309"
    return "匹配度低", "#9ca3af"


def role_tag(bp: str, role: dict) -> str:
    if bp not in role:
        return '<span class="tag none">其他项目</span>'
    name, cls = ROLE_BADGE[role[bp]]
    return f'<span class="tag {cls}">{"✓" if cls == "ok" else "⚠"} {name}</span>'


challenges = load_challenges()
answers = load_json(str(ANSWERS_PATH))["challenges"]
metrics = load_metrics()
baseline = load_json(str(MATCH_DIR / "baseline_tfidf.json"))

# ---------- 顶部 ----------
st.title("🧬 企业难题 × BP 智能匹配")
# 用 HTML 加粗：Markdown 的 ** 紧挨中文标点时不会被识别成加粗
st.markdown("用大模型把<b>企业技术难题</b>和<b>参赛团队的商业计划书（BP）</b>做智能匹配："
            "先结构化理解双方，再粗筛、精排，给出有原文证据、有风险提示的推荐。", unsafe_allow_html=True)
st.info("**真实业务数据不公开**。本页面使用与真实数据格式一致的**合成数据**（synthetic）演示："
        "30 份 BP、5 道企业难题由大模型合成，其中的机构、人物与数值均不对应任何真实主体；"
        "结果为预先运行生成，页面不调用任何 API。", icon="ℹ️")

steps = [("📄 解析", "PDF → 文字", ""), ("🧩 结构化抽取", "DeepSeek 抽取字段<br>原文/推测分开标注", "llm"),
         ("🔎 粗筛", "标签重合 + TF-IDF<br>不调用大模型，取前 15", ""),
         ("⚖️ 精排", "DeepSeek 按 100 分标准打分<br>引用原文、提示风险", "llm"),
         ("📊 评测", "对照标准答案<br>和关键词基线比较", "")]
flow = '<span class="arrow">→</span>'.join(
    f'<div class="step {cls}"><b>{name}</b><span>{desc}</span></div>' for name, desc, cls in steps)
st.markdown(f'<div class="flow">{flow}</div>', unsafe_allow_html=True)

# ---------- 选择难题 ----------
st.divider()
cid = st.selectbox(
    "选择一道企业难题", list(challenges),
    format_func=lambda c: f"{c} · {challenges[c]['company']} · {challenges[c]['core_problem']['value'][:28]}…")
ch = challenges[cid]
summary = load_json(str(MATCH_DIR / f"{cid}.json"))
role = metrics["per_challenge"][cid]["role"]

# ---------- 结构化拆解 ----------
st.subheader("① 难题拆解")
st.markdown('<span class="tag orig">原文</span> 题目里直接写到的　'
            '<span class="tag inf">推测</span> 系统推断的（匹配时只算一半分）　'
            '<span style="opacity:.6;font-size:.85rem">鼠标停在标签上可以看依据</span>',
            unsafe_allow_html=True)
left, right = st.columns([5, 4], gap="large")
with left:
    st.markdown(f'<div class="quote"><b>{esc(ch["company"])}</b>（{esc(ch["direction"])}）<br>{esc(ch["title"])}</div>',
                unsafe_allow_html=True)
    if ch["removed_content"]:
        st.markdown("已去掉的政策背景 / 非技术表述：<br>" + "<br>".join(
            f'<span class="removed">{esc(x)}</span>' for x in ch["removed_content"]), unsafe_allow_html=True)
with right:
    rows = [("核心问题", tags(ch["core_problem"])), ("补充理解", tags(ch["core_problem_inferred"])),
            ("工艺环节", tags(ch["process_stages"])), ("应用领域", tags(ch["application_categories"])),
            ("目标产物", tags(ch["target_products"])), ("技术关键词", tags(ch["tech_keywords"])),
            ("最低成熟度", tags(ch["min_maturity"]))]
    st.markdown("".join(f'<div class="row"><div class="k">{k}</div><div>{v}</div></div>' for k, v in rows),
                unsafe_allow_html=True)

# ---------- 前 3 名推荐 ----------
st.subheader("② 系统推荐 Top 3")
st.caption(f"粗筛从 30 份 BP 里选出 15 个候选，大模型逐个打分。总分 = 各项得分 − 推测扣分；"
           f"75 分以上推荐对接，{RECOMMEND_SCORE}–74 分可以考虑。")
for r in summary["top"]:
    with st.container(border=True):
        c1, c2 = st.columns([1, 3], gap="large")
        name, color = level(r["total"])
        with c1:
            st.markdown(f'<div style="opacity:.6">第 {r["final_rank"]} 名</div>'
                        f'<div class="score" style="color:{color}">{r["total"]}<small> / 100</small></div>'
                        f'<div style="color:{color};font-weight:600;margin-top:.3rem">{name}</div>',
                        unsafe_allow_html=True)
            if r["inference_deducted"]:
                st.markdown(f'<span class="tag inf">推测扣分 −{r["inference_deducted"]}</span>',
                            unsafe_allow_html=True)
        with c2:
            st.markdown(f'**{esc(r["project_name"])}**<br><span style="opacity:.65">{esc(r["company_name"])}</span>',
                        unsafe_allow_html=True)
            bars = "".join(
                f'<div class="bar-label"><span>{label}</span><span>{r[k]["score"]} / {full}</span></div>'
                f'<div class="bar"><div style="width:{r[k]["score"] / full:.0%}"></div></div>'
                for k, (label, full) in CRITERIA.items())
            st.markdown(bars, unsafe_allow_html=True)
        st.markdown(f"**推荐理由**：{r['recommendation']}")
        e1, e2 = st.columns(2, gap="large")
        with e1:
            st.markdown("**BP 原文证据**")
            st.markdown("".join(f'<div class="quote">{esc(q)}</div>' for q in r["evidence_quotes"]),
                        unsafe_allow_html=True)
        with e2:
            st.markdown("**差距或风险**")
            st.markdown("\n".join(f"- {g}" for g in r["gaps_risks"]))
            for d in r["inference_deductions"]:
                st.caption(f"⚠️ {CRITERIA[d['criterion']][0]} −{d['points']}：{d['reason']}")
        with st.expander("查看 BP 原文"):
            st.text((BP_DIR / f"{r['bp']}.txt").read_text(encoding="utf-8"))

# ---------- 对比区 ----------
st.subheader("③ 对比：关键词基线 vs 完整系统")
st.caption("关键词基线只按 TF-IDF 文本相似度排名。标签说明：✓ 正确答案；陷阱a = 关键词很像但解决不了；"
           "陷阱b = 关键词不同但能解决（也是正确答案）。")
b_col, s_col = st.columns(2, gap="large")


def rank_item(rank: int, bp: str, title: str, meta: str) -> str:
    cls = "" if bp not in role else ("bad" if role[bp] == "decoy" else "ok")
    return (f'<div class="rank-item {cls}"><b>{rank}. {esc(title)}</b><br>'
            f'{role_tag(bp, role)}<span class="meta">{meta}</span></div>')


with b_col:
    st.markdown("**关键词基线（TF-IDF）前 3**")
    st.markdown("".join(rank_item(b["text_rank"], b["bp"], b["project_name"], f'文本相似度 {b["text_score"]:.3f}')
                        for b in baseline[cid][:K]), unsafe_allow_html=True)
with s_col:
    st.markdown("**完整系统（粗筛 + 精排）前 3**")
    st.markdown("".join(rank_item(r["final_rank"], r["bp"], r["project_name"], f'总分 {r["total"]}')
                        for r in summary["top"]), unsafe_allow_html=True)

ranks, score = metrics["per_challenge"][cid]["ranks"], metrics["per_challenge"][cid]["score"]
ans = answers[cid]
trap_rows = []
for item, label in [(ans["trap_keyword_decoy"], "陷阱a（不应推荐）"),
                    *[(g, "陷阱b（应推荐）") for g in ans["gold"] if g["type"] == "gold_hidden"]]:
    bp = item["bp"]
    s_rank = ranks["system"].get(bp)
    trap_rows.append({"陷阱": label, "项目": bp.split("_", 1)[1],
                      "关键词基线名次": ranks["baseline"].get(bp),
                      "完整系统名次": s_rank if s_rank else "未进入粗筛",
                      "系统得分": score.get(bp, "-"), "设计说明": item["why"]})
st.markdown("**这道题的两个陷阱**")
st.dataframe(pd.DataFrame(trap_rows), hide_index=True, width="stretch",
             column_config={"设计说明": st.column_config.TextColumn(width="large")})

# ---------- 评测结果 ----------
st.subheader("④ 评测结果（5 道题汇总）")
n, b, s, rec = metrics["n"], metrics["totals"]["baseline"], metrics["totals"]["system"], metrics["recommended"]


def frac(x) -> str:
    return f"{x}/{n}" if isinstance(x, int) else f"{x / n:.0%}"


m1, m2, m3 = st.columns(3)
m1.metric("陷阱a 被识别（没进前 3）", frac(s["decoy_out"]), f"基线 {frac(b['decoy_out'])}")
m2.metric("Top-3 召回率", frac(s["recall"]), f"基线 {frac(b['recall'])}")
m3.metric(f"陷阱b 达到推荐线（≥{RECOMMEND_SCORE} 分）", frac(rec["gold_hidden"]),
          f"陷阱a 被误推荐 {frac(rec['decoy'])}", delta_color="off")

table = pd.DataFrame([
    ("Top-3 命中率（前 3 里至少 1 个正确答案）", frac(b["hit"]), frac(s["hit"])),
    ("Top-3 召回率（2 个正确答案平均找回几成）", frac(b["recall"]), frac(s["recall"])),
    ("Top-1 准确率（第 1 名就是正确答案）", frac(b["top1"]), frac(s["top1"])),
    ("陷阱b 被识别（关键词不同但能解决，进了前 3）", frac(b["hidden"]), frac(s["hidden"])),
    ("陷阱a 被识别（关键词像但解决不了，没进前 3）", frac(b["decoy_out"]), frac(s["decoy_out"])),
], columns=["指标", "关键词基线", "完整系统"])
st.dataframe(table, hide_index=True, width="stretch")
st.caption("说明：样本只有 5 道题、30 份 BP，单次运行；评测集偏简单，Top-3 命中率两种方法都是 100%，"
           "主要差别在于能否排除关键词像但解决不了的项目。精排打分有随机波动，相差 5 分以内的名次不应视为有明确先后。"
           "完整方法和局限见 docs/evaluation.md。")
