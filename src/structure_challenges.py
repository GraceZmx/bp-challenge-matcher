"""用 DeepSeek 把企业难题（揭榜挂帅赛题）拆解成结构化 JSON。

工艺环节、应用领域、成熟度与 BP 使用完全相同的清单，方便后续匹配。
每条内容都标注来源：原文（附原话，程序会核对）或推测（附理由）。

结果：output/challenges_structured/<编号>.json，汇总 output/challenges_table.csv
用法：.venv/bin/python src/structure_challenges.py
"""
import csv
import json
import sys
from datetime import datetime

import openpyxl

from extract_structured import (MATURITY_OPTIONS, MODEL, NA, PROCESS_STAGE_OPTIONS,
                                ROOT, load_client, squash)
from paths import CHALLENGE_OVERRIDES, CHALLENGE_XLSX, OUTPUT_DIR

XLSX_PATH = CHALLENGE_XLSX
OUT_DIR = OUTPUT_DIR / "challenges_structured"
CSV_PATH = OUTPUT_DIR / "challenges_table.csv"
ERROR_LOG = OUTPUT_DIR / "challenges_errors.log"
# 人工修正：模型抽取后，用这里的内容覆盖对应字段（重跑也会保留）；合成数据没有人工修正
OVERRIDES_PATH = CHALLENGE_OVERRIDES
CATEGORY_OPTIONS = list(json.loads(
    (ROOT / "config" / "application_field_map.json").read_text(encoding="utf-8")))

SOURCES = ["原文", "推测"]
SINGLE_FIELDS = ["core_problem", "core_problem_inferred", "min_maturity"]
LIST_FIELDS = ["process_stages", "application_categories", "target_products", "tech_keywords"]
# removed_content 放在最前面：让模型先想清楚删掉什么，再写核心问题
FIELDS = (["removed_content"] + SINGLE_FIELDS[:2] + LIST_FIELDS[:3]
          + SINGLE_FIELDS[2:] + LIST_FIELDS[3:])
# 这些字段标"原文"时，value 本身必须是题目里出现过的词
VERBATIM_FIELDS = ["target_products", "tech_keywords"]

CATEGORY_DEFINITIONS = """  - 医药健康：药品、疫苗、医疗器械、诊断、医美等医疗健康用途
  - 食品营养：食品、饮料、营养品、保健食品及其原料
  - 美妆日化：化妆品、个人护理、日化用品及其原料
  - 农业养殖：种植、种业、农药肥料、饲料、水产畜牧
  - 化工材料：化学品、材料、染料、纺织、包装等工业产品
  - 能源环保：能源燃料、废水废气处理、碳减排
  - 生物制造与科研服务：只有当目标产物本身就是研发平台、设备、软件、科研试剂或技术服务时才选。
  注意：
  - 应用领域指的是出题企业的目标产物最终用在哪个行业，可以结合产物和出题企业名称推测
  - 出题企业想找技术来改进自己的生产，不等于属于"生物制造与科研服务"，应按它的产物用途归类
  - 产物是用生物方法生产的（如发酵生产的原料），也不等于属于这一类"""
# 核心问题里出现这些词，说明非技术诉求没去干净
NON_TECH_TERMS = ["市场", "推广", "品牌"]

SYSTEM_PROMPT = f"""你是生物制造领域的技术需求分析师，负责把企业发布的技术难题拆解成结构化数据，用于和参赛项目匹配。

【规则】
1. 只提炼真正的技术需求。以下内容一律忽略：
   - 政策背景、战略意义、口号式表述（如"紧扣某某规划纲要""助力从大国到强国"）
   - 非技术诉求，如"市场效果""市场推广""品牌宣传"
2. 每一条内容都要标注来源，原文和推测严格分开：
   - "原文"：题目原文直接写到的。basis 必须一字不差地照抄题目里支撑它的片段。
     只要在原文基础上增加了意思（哪怕是合理的解读），就必须标"推测"。
   - "推测"：原文没有直接写，但能合理推断的。basis 用一句话写推测理由。
   原文没写、也无法合理推测的，就不要写；列表字段可以是空列表。
3. 只有"题目"算原文。"出题企业"和"所属方向"（大赛的赛道分类）可以作为推测的依据，但不能标"原文"。

【应用领域类别定义】
{CATEGORY_DEFINITIONS}

【输出格式】只输出一个 JSON 对象。字段如下：
- removed_content（字符串列表，第一个输出）：先逐条列出题目里要删掉的政策背景和非技术诉求，每条照抄原文；没有就填空列表
以下字段每一条内容的格式都是 {{"value": ..., "source": "原文"或"推测", "basis": ...}}：
- core_problem（单条，source 必须是"原文"）：核心技术问题。value 只能从题目原文里删字得到（删掉政策背景和非技术诉求），不能增加或改写任何字。
  例：题目"X的产率提升及品牌推广" → 删成"X的产率提升"
- core_problem_inferred（单条或 null，source 必须是"推测"）：对核心问题的补充理解（如具体要解决什么瓶颈）；没有可补充的就填 null
- process_stages（列表）：涉及的工艺环节，value 只能从这个清单选：{PROCESS_STAGE_OPTIONS}
- application_categories（列表）：应用领域，value 只能从上面 7 个类别里选
- target_products（列表）：目标产物或产品。标"原文"的，value 必须是题目里原样出现的词
- min_maturity（单条）：对解决方案成熟度的最低要求，value 只能从 {MATURITY_OPTIONS} 中选；
  原文没写且无法推测时，value 填"{NA}"，source 填"推测"，basis 说明原因
- tech_keywords（列表）：技术关键词，3-8 个，用于检索匹配。标"原文"的，value 必须是题目里原样出现的词"""


def read_challenges() -> list[dict]:
    ws = openpyxl.load_workbook(XLSX_PATH, read_only=True).active
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    col = {name: header.index(name) for name in ("出题企业", "题目", "所属方向")}
    challenges = []
    for row in rows:
        if not row[col["题目"]]:
            continue
        challenges.append({
            "id": f"C{len(challenges) + 1:02d}",
            "company": str(row[col["出题企业"]]).strip(),
            "title": str(row[col["题目"]]).strip(),
            "direction": str(row[col["所属方向"]] or "").strip(),
        })
    return challenges


def source_text(ch: dict) -> str:
    return f"出题企业：{ch['company']}\n所属方向：{ch['direction']}\n题目：{ch['title']}"


def is_deletion_of(short: str, long: str) -> bool:
    """short 是否只是从 long 里删掉一些字得到的（不增加、不改写、不调换顺序）"""
    chars = iter(long)
    return all(c in chars for c in short)


def check_item(item, field: str, title: str, options: list[str] | None):
    """title 是题目原文；只有题目算原文，所属方向和企业名不算"""
    if not isinstance(item, dict) or not all(k in item for k in ("value", "source", "basis")):
        raise ValueError(f"{field} 的条目格式不对：{item}")
    if item["source"] not in SOURCES:
        raise ValueError(f"{field} 的 source 只能是原文/推测：{item['source']}")
    if options and item["value"] not in options:
        raise ValueError(f"{field} 的 value 不在清单内：{item['value']}")
    if item["source"] != "原文":
        return
    if squash(item["basis"]) not in squash(title):
        raise ValueError(f"{field} 标为原文，但 basis 在题目中找不到：{item['basis']}")
    if field in VERBATIM_FIELDS and squash(item["value"]) not in squash(title):
        raise ValueError(f"{field} 标为原文，但 value 不是题目里的原词：{item['value']}")


def validate(data: dict, title: str) -> dict:
    missing = [f for f in FIELDS if f not in data]
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    options = {"process_stages": PROCESS_STAGE_OPTIONS,
               "application_categories": CATEGORY_OPTIONS,
               "min_maturity": MATURITY_OPTIONS + [NA]}

    removed = data["removed_content"]
    if not isinstance(removed, list) or not all(isinstance(x, str) for x in removed):
        raise ValueError("removed_content 应为字符串列表")
    not_found = [x for x in removed if squash(x) not in squash(title)]
    if not_found:
        raise ValueError(f"removed_content 在题目中找不到：{not_found}")

    core = data["core_problem"]
    check_item(core, "core_problem", title, None)
    if core["source"] != "原文" or not is_deletion_of(squash(core["value"]), squash(title)):
        raise ValueError(f"core_problem 只能从题目里删字得到：{core['value']}")
    leftover = [t for t in NON_TECH_TERMS if t in core["value"]]
    if leftover:
        raise ValueError(f"core_problem 里还有非技术诉求：{leftover}")
    inferred = data["core_problem_inferred"]
    if inferred is not None:
        check_item(inferred, "core_problem_inferred", title, None)
        if inferred["source"] != "推测":
            raise ValueError("core_problem_inferred 的 source 必须是推测")
    check_item(data["min_maturity"], "min_maturity", title, options["min_maturity"])

    for f in LIST_FIELDS:
        if not isinstance(data[f], list):
            raise ValueError(f"{f} 应为列表")
        for item in data[f]:
            check_item(item, f, title, options.get(f))
    return {f: data[f] for f in FIELDS}


def extract_one(client, ch: dict, last_err: str | None = None) -> dict:
    """last_err：上一次没通过检查的原因，重试时告诉模型要改什么"""
    prompt = f"题目原文如下：\n\n{source_text(ch)}"
    if last_err:
        prompt += f"\n\n注意：你上一次的输出没有通过检查，原因是：{last_err}\n请修正后重新输出完整的 JSON。"
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return validate(json.loads(resp.choices[0].message.content), ch["title"])


def fmt(items) -> str:
    """表格里显示用：推测的内容加上（推测）标记"""
    if items is None:
        return ""
    if isinstance(items, dict):
        items = [items]
    return "；".join(f"{x['value']}{'（推测）' if x['source'] == '推测' else ''}"
                    for x in items) or NA


def apply_overrides():
    """把人工修正写进已有的 json；可以重复执行，结果不变"""
    if not OVERRIDES_PATH or not OVERRIDES_PATH.exists():
        return
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    for cid, fields in overrides.items():
        path = OUT_DIR / f"{cid}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        changes = {k: v for k, v in fields.items() if not k.startswith("_")}
        data.update(changes)
        data["manual_overrides"] = {"fields": list(changes), "note": fields.get("_note", "")}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"✏️  {cid} 已套用人工修正：{list(changes)}")


def build_csv() -> int:
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append({"id": d["id"], "company": d["company"], "direction": d["direction"],
                     **{k: fmt(d[k]) for k in FIELDS if k != "removed_content"},
                     "removed_content": "；".join(d["removed_content"]), "title": d["title"]})
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["id", "company", "direction", *FIELDS, "title"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    challenges = read_challenges()
    todo = [c for c in challenges if not (OUT_DIR / f"{c['id']}.json").exists()]
    print(f"共 {len(challenges)} 道题，已处理过 {len(challenges) - len(todo)} 道（跳过），本次处理 {len(todo)} 道")

    client = load_client()
    errors = []
    for ch in todo:
        last_err = None
        for _ in range(2):  # 第一次 + 重试一次
            try:
                data = {**ch, **extract_one(client, ch, last_err)}
                (OUT_DIR / f"{ch['id']}.json").write_text(
                    json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                last_err = None
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                print(f"   ↻ {ch['id']} 未通过检查：{last_err}")
        print(f"{'❌' if last_err else '✅'} {ch['id']} {ch['company']}")
        if last_err:
            errors.append((ch["id"], last_err))

    if errors:
        with ERROR_LOG.open("a", encoding="utf-8") as fp:
            for cid, err in errors:
                fp.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{cid}\t{err}\n")
    apply_overrides()
    print(f"\n本次成功 {len(todo) - len(errors)} 道，失败 {len(errors)} 道；汇总表共 {build_csv()} 行")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
