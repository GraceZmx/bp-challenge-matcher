"""生成合成评测集（可公开）：30 份 BP 项目简介 + 5 道企业难题 + 标准答案。

每道题设计 3 个项目：
  gold_obvious：明显能解决（关键词也对得上）
  gold_hidden ：陷阱(b) 关键词不一样，但其实能解决
  decoy       ：陷阱(a) 关键词很像，但其实解决不了
另有 15 个干扰项目，不能解决任何一道题。

输出（系统运行时只读前两个，标准答案只给 eval/evaluate.py 用）：
  demo_data/bps/D01_xxx.txt …    30 份 BP（随机编号，不透露角色）
  demo_data/challenges.xlsx      5 道企业难题
  demo_data/answers.json         标准答案和设计说明

用法：.venv/bin/python eval/generate_demo_data.py
"""
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import openpyxl
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = ROOT / "demo_data"
BP_DIR = DEMO_DIR / "bps"
XLSX_PATH = DEMO_DIR / "challenges.xlsx"
ANSWERS_PATH = DEMO_DIR / "answers.json"
SPECS_CACHE = DEMO_DIR / ".specs_cache.json"   # 生成中间结果，中断后可以接着跑
REAL_TEXT_DIR = ROOT / "output" / "texts"       # 只用于检查有没有照抄真实 BP，不会写进合成数据
REAL_STRUCT_DIR = ROOT / "output" / "structured"

MODEL = "deepseek-chat"
SEED = 20261022
N_DISTRACTORS = 15
MAX_TRIES = 3
SYNTHETIC_HEADER = ("【SYNTHETIC｜合成数据】本文件为大模型按真实数据格式生成的合成数据，用于在不公开真实业务数据的前提下"
                    "演示和评测系统；文中机构、人物与数值均不对应任何真实主体。")
DIRECTIONS = ["新型基因编辑与细胞工厂构建方向", "酶工程技术方向", "生物制造关键设备及试剂国产化方向",
              "食品及日化功能原料方向", "绿色生物制造技术方向", "AI驱动的生物系统设计与数字化生物工艺平台方向",
              "生命健康方向"]
# 5 道题分别落在不同领域，避免互相干扰
SCENARIO_THEMES = ["食品配料的生产工艺", "农业或饲料用微生物制剂", "化妆品功效原料",
                   "工业酶或化工材料的生物制造", "发酵过程的检测或控制设备"]
# 人工审核后的改名：生成的名称和真实企业、品牌太像，或者两道题的出题企业重名（2026-10-22 审核）
MANUAL_RENAMES = {
    "杭州蓝晶合成生物科技有限公司": "杭州砚序合成生物科技有限公司",  # 像真实企业"蓝晶微生物"
    "蓝晶合成生物技术有限公司": "霜启合成生物技术有限公司",
    "微康生物工程有限公司": "暖芽生物工程有限公司",                # 像真实益生菌企业"微康"
    "微谱科学仪器有限公司": "棱光科学仪器有限公司",                # 像真实检测企业"微谱"
    "绿源生物科技有限公司": "青牧生物科技有限公司",                # C02、C05 出题企业重名
    "绿源生物工程有限公司": "衡酵生物工程有限公司",
}


def rename(text: str) -> str:
    for old, new in MANUAL_RENAMES.items():  # 长名字排在前面，避免被短名字先替换
        text = text.replace(old, new)
    return text


# 太笼统的词（哪份 BP 都可能用到），不作为陷阱的用词要求
GENERIC_TERMS = {"生物制造", "高纯度", "高稳定性", "高效", "稳定性"}
FORBIDDEN_WORDS = ["陷阱", "标准答案", "揭榜", "赛题", "出题", "匹配", "decoy", "trap", "SYNTHETIC_ROLE"]

FICTION_RULE = ("所有公司名、人名、数据都必须是合成的，不要使用任何真实存在的企业、品牌、高校、科研机构或人物名称；"
                "高校、研究院等机构名也不要用真实地名开头（如\"江南\"\"东海\"\"北方\"），以免和真实机构混淆。")
# 机构名里出现这些真实地名，说明可能和真实高校、研究院混淆，生成时判为不合格（2026-10-22 人工审核后补充）
REAL_PLACE_PREFIXES = ["江南", "东海", "北方", "华东", "华南", "华北", "西北", "西南", "中南", "东北"]

SCENARIO_PROMPT = """你在为一个"企业技术难题 × 参赛项目"智能匹配系统设计评测数据。{fiction}

请围绕主题"{theme}"设计 1 道企业技术难题，以及 3 个参赛项目：
- gold_obvious：能直接解决这道难题，描述里自然会用到难题里的关键技术词
- gold_hidden：同样能真正解决难题的核心需求，但它来自另一个技术圈子，用完全不同的术语描述自己
  （例如难题说"提高制剂稳定性"，它说的是"微胶囊包埋/冻干保护剂配方"），不能使用难题里的关键技术词
- decoy：和难题的关键词高度重合（同样的产物名、同样的工艺名词），但因为一个具体原因其实解决不了
  （例如它只做检测不做生产、只在毫升级实验、针对的是另一种剂型或另一个环节）

要求：
- 难题的"题目"写 1-2 句话，风格像企业发布的技术需求；{boilerplate}
- 难题必须有明确、具体的技术瓶颈，否则无法判断谁能解决
- 3 个项目的公司名都和出题企业不同

只输出一个 JSON：
{{
  "challenge": {{"company": "出题企业", "title": "题目", "direction": "从 {directions} 中选一个",
                "key_terms": ["难题里的 3-5 个关键技术词"]}},
  "gold_obvious": {{"project_name": "...", "company_name": "...", "brief": "项目要点 3-5 句", "why": "为什么能解决"}},
  "gold_hidden": {{"project_name": "...", "company_name": "...", "brief": "...", "why": "为什么能解决，以及它用的是哪些不同的术语"}},
  "decoy": {{"project_name": "...", "company_name": "...", "brief": "...", "why": "为什么关键词像但解决不了"}}
}}"""

DISTRACTOR_PROMPT = """你在为一个生物制造项目匹配系统设计评测数据。{fiction}

请设计 {n} 个生物制造领域的参赛项目，覆盖医药、食品、美妆、农业、材料、能源环保、科研仪器等不同方向，
彼此不要重复。它们不能解决下面任何一道企业难题（主题要明显不同）：
{challenges}

只输出一个 JSON：{{"projects": [{{"project_name": "...", "company_name": "...", "brief": "项目要点 3-5 句"}}, ...]}}"""

BP_PROMPT = """请把下面的项目要点写成一份"参赛项目简介"。{fiction}

【项目要点】
项目名称：{project_name}
公司名称：{company_name}
要点：{brief}
{vocab_rule}

【格式】严格按下面的结构输出纯文本，总长 700-1000 字，风格简洁、像评审整理过的材料：
生物制造创新大赛（合成数据）决赛项目简介
项目名称：…
公司名称：…
赛道及方向：应用创新赛道 / …方向
公司简介
（2-3 句：成立时间、核心团队带头人的姓名和背景，均为合成）
01 技术对象与应用场景
02 技术原理与工艺路线
核心原理：…
工艺路线：A→B→C→D
03 量化指标 数据 对照与证据
指标
原稿数据 / 目标
对照条件与待核实口径
（3 个指标，每个指标 3 行：指标名 / 原稿数据或目标 / 还需补充或核实什么）
04 产品形态与服务对象
05 产业化状态与计划节点
（必须有一句"报名阶段为实验室/小试/中试/产品已上市"之一）
06 落地需求
07 融资需求
复赛得分
（70-85 之间的两位小数）分

【禁止】不要提到任何企业难题、匹配、评测；只写这个项目本身。"""

REQUIRED_SECTIONS = ["项目名称：", "公司名称：", "赛道及方向：", "01 技术对象", "02 技术原理",
                     "03 量化指标", "04 产品形态", "05 产业化", "06 落地需求", "07 融资需求", "复赛得分"]


def load_client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        sys.exit("❌ .env 里的 DEEPSEEK_API_KEY 是空的")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=180)


def squash(s: str) -> str:
    return "".join(s.split())


def chat(client, prompt: str, json_mode: bool) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} if json_mode else None,
        temperature=0.8,  # 生成数据要有变化，和系统运行时的 0 不同
    )
    return resp.choices[0].message.content


class RealDataGuard:
    """检查合成内容有没有照抄真实 BP：真实公司名、项目名，以及 20 字以上的相同片段。

    比对前先按格式标签切开（标签和前后内容拼在一起不算照抄），出现在 3 份以上真实 BP 里的片段算套话。
    """
    WINDOW = 20
    LABELS = re.compile("|".join(re.escape(squash(x)) for x in REQUIRED_SECTIONS + [
        "决赛项目简介", "公司简介", "核心原理：", "工艺路线：", "指标", "原稿数据/目标", "对照条件与待核实口径",
        "01 技术对象与应用场景", "02 技术原理与工艺路线", "03 量化指标 数据 对照与证据",
        "04 产品形态与服务对象", "05 产业化状态与计划节点", "报名阶段为",
        "意向落地区域：", "融资轮次/金额：", "意向资本：", "投后估值："]))

    def __init__(self):
        self.names = []
        for f in REAL_STRUCT_DIR.glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            # 项目名太短的（如"麦角硫因"）只是化合物名，不算泄露
            self.names += [d["company_name"]] + ([d["project_name"]] if len(d["project_name"]) >= 8 else [])
        df = Counter()
        for f in REAL_TEXT_DIR.glob("*.txt"):
            df.update(self._windows(f.read_text(encoding="utf-8")))
        # 出现在 3 份以上真实 BP 里的片段算模板套话，不算照抄
        self.real_windows = {w for w, n in df.items() if n < 3}

    def _windows(self, text: str) -> set[str]:
        out = set()
        for seg in self.LABELS.split(squash(text)):
            out |= {seg[i:i + self.WINDOW] for i in range(len(seg) - self.WINDOW + 1)}
        return out

    def problems(self, text: str) -> list[str]:
        found = [n for n in self.names if n in text]
        copied = self._windows(text) & self.real_windows
        out = [f"出现了真实名称：{found}"] if found else []
        if copied:
            out.append(f"和真实 BP 有相同片段：{sorted(copied)[:3]}")
        return out


def design_scenarios(client, guard, cache) -> list[dict]:
    scenarios = cache.setdefault("scenarios", [])
    for i, theme in enumerate(SCENARIO_THEMES[len(scenarios):], len(scenarios)):
        # 第 3 道题故意加政策背景，测试系统能不能去掉套话
        boilerplate = ("请在技术需求前后加上一两句政策背景和口号式表述（如响应国家战略、实现自主可控）"
                       if i == 2 else "只写技术需求，不要政策背景")
        for attempt in range(MAX_TRIES):
            s = json.loads(chat(client, SCENARIO_PROMPT.format(
                theme=theme, boilerplate=boilerplate, directions=DIRECTIONS, fiction=FICTION_RULE), json_mode=True))
            text = json.dumps(s, ensure_ascii=False)
            problems = guard.problems(text)
            companies = [s["challenge"]["company"]] + [s[r]["company_name"]
                                                       for r in ("gold_obvious", "gold_hidden", "decoy")]
            if len(set(companies)) < 4:
                problems.append("项目公司名和出题企业重复")
            if not problems:
                break
            print(f"   ↻ 场景 {i + 1} 未通过检查：{problems}")
        else:
            sys.exit(f"❌ 场景 {i + 1} 连续 {MAX_TRIES} 次未通过检查")
        scenarios.append(s)
        SPECS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 场景 {i + 1}：{s['challenge']['company']}｜{s['challenge']['title'][:40]}")
    return scenarios


def design_distractors(client, guard, cache, scenarios) -> list[dict]:
    if "distractors" not in cache:
        titles = "\n".join(f"- {s['challenge']['title']}" for s in scenarios)
        data = json.loads(chat(client, DISTRACTOR_PROMPT.format(n=N_DISTRACTORS, challenges=titles, fiction=FICTION_RULE),
                               json_mode=True))
        projects = data["projects"][:N_DISTRACTORS]
        if len(projects) < N_DISTRACTORS:
            sys.exit(f"❌ 干扰项目只生成了 {len(projects)} 个")
        cache["distractors"] = projects
        SPECS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✅ 干扰项目 {N_DISTRACTORS} 个")
    return cache["distractors"]


def check_bp(text: str, spec: dict, guard, avoid_names: list[str]) -> list[str]:
    problems = [f"缺少段落：{s}" for s in REQUIRED_SECTIONS if s not in text]
    problems += [f"出现了禁用词：{w}" for w in FORBIDDEN_WORDS if w in text]
    problems += [f"出现了出题企业名：{n}" for n in avoid_names if n in text]
    problems += [f"机构名用了真实地名开头：{m.group(0)}" for m in re.finditer(
        "(" + "|".join(REAL_PLACE_PREFIXES) + r")[\u4e00-\u9fff]{0,10}(大学|学院|研究院|研究所|中心)", text)]
    if not re.search(r"报名阶段为(实验室|小试|中试|产品已上市)", text):
        problems.append('缺少"报名阶段为…"')
    body = text.split("\n", 1)[-1]  # 第一行是统一的大赛标题，不检查用词
    if spec.get("avoid_terms"):
        used = [t for t in spec["avoid_terms"] if t in body]
        if used:
            problems.append(f"用了不该用的难题关键词：{used}")
    if spec.get("require_terms"):
        used = [t for t in spec["require_terms"] if t in body]
        if len(used) < min(2, len(spec["require_terms"])):
            problems.append(f"难题关键词用得太少，只用了：{used}")
    return problems + guard.problems(text)


def write_bp(client, spec: dict, guard, avoid_names) -> str:
    vocab_rule = ""
    if spec.get("avoid_terms"):
        vocab_rule = f"【用词要求】全文不能出现这些词：{spec['avoid_terms']}，请用你自己领域的术语描述。"
    elif spec.get("require_terms"):
        vocab_rule = f"【用词要求】请在介绍中自然地使用这些词：{spec['require_terms']}。"
    prompt = BP_PROMPT.format(fiction=FICTION_RULE, project_name=spec["project_name"], company_name=spec["company_name"],
                              brief=spec["brief"], vocab_rule=vocab_rule)
    feedback = ""
    for _ in range(MAX_TRIES):
        text = chat(client, prompt + feedback, json_mode=False).strip()
        problems = check_bp(text, spec, guard, avoid_names)
        if not problems:
            return text
        print(f"   ↻ {spec['project_name'][:20]} 未通过检查：{problems}")
        feedback = f"\n\n注意：你上一次写的没有通过检查，原因是：{problems}。请修正后重写全文。"
    sys.exit(f"❌ {spec['project_name']} 连续 {MAX_TRIES} 次未通过检查")


def safe_name(s: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "", s)[:40]


def main():
    DEMO_DIR.mkdir(exist_ok=True)
    BP_DIR.mkdir(exist_ok=True)
    cache = json.loads(SPECS_CACHE.read_text(encoding="utf-8")) if SPECS_CACHE.exists() else {}
    client, guard = load_client(), RealDataGuard()

    scenarios = design_scenarios(client, guard, cache)
    distractors = design_distractors(client, guard, cache, scenarios)

    # 汇总全部 30 个项目的说明书
    specs = []
    for ci, s in enumerate(scenarios, 1):
        cid = f"C{ci:02d}"
        terms = [t for t in s["challenge"]["key_terms"] if t not in GENERIC_TERMS]
        specs.append({**s["gold_obvious"], "role": "gold_obvious", "challenge": cid})
        specs.append({**s["gold_hidden"], "role": "gold_hidden", "challenge": cid, "avoid_terms": terms})
        specs.append({**s["decoy"], "role": "decoy", "challenge": cid, "require_terms": terms})
    specs += [{**d, "role": "distractor", "challenge": None} for d in distractors]

    # 随机编号，文件名和顺序都不透露角色
    random.Random(SEED).shuffle(specs)
    texts = cache.setdefault("texts", {})
    avoid_names = [s["challenge"]["company"] for s in scenarios]
    for i, spec in enumerate(specs, 1):
        spec["bp"] = f"D{i:02d}_{safe_name(spec['project_name'])}"
        if spec["bp"] not in texts:
            texts[spec["bp"]] = write_bp(client, spec, guard, avoid_names)
            SPECS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✅ [{i}/{len(specs)}] {spec['bp']}")

    for old in BP_DIR.glob("*.txt"):
        old.unlink()
    for spec in specs:
        (BP_DIR / f"{spec['bp']}.txt").write_text(
            f"{SYNTHETIC_HEADER}\n{rename(texts[spec['bp']])}\n", encoding="utf-8")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "企业赛题汇总(synthetic)"
    ws.append(["出题企业", "题目", "所属方向", "数据说明"])
    for s in scenarios:
        c = s["challenge"]
        ws.append([rename(c["company"]), rename(c["title"]), c["direction"], "SYNTHETIC 合成数据"])
    wb.save(XLSX_PATH)

    answers = {"synthetic": True, "manual_renames": MANUAL_RENAMES,
               "note": "合成评测集的标准答案。只给 eval/evaluate.py 使用，系统运行时不读取本文件。",
               "seed": SEED, "challenges": {}}
    for ci, s in enumerate(scenarios, 1):
        cid = f"C{ci:02d}"
        role_bp = {sp["role"]: sp for sp in specs if sp["challenge"] == cid}
        answers["challenges"][cid] = {
            "company": rename(s["challenge"]["company"]),
            "title": rename(s["challenge"]["title"]),
            "key_terms": s["challenge"]["key_terms"],
            "gold": [{"bp": role_bp[r]["bp"], "type": r, "why": rename(role_bp[r]["why"])}
                     for r in ("gold_obvious", "gold_hidden")],
            "trap_keyword_decoy": {"bp": role_bp["decoy"]["bp"], "why": rename(role_bp["decoy"]["why"])},
        }
    answers["distractors"] = [sp["bp"] for sp in specs if sp["role"] == "distractor"]
    ANSWERS_PATH.write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已生成 {len(specs)} 份 BP、{len(scenarios)} 道题，标准答案在 {ANSWERS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
