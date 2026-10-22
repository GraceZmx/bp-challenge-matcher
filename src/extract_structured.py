"""用 DeepSeek 把 output/texts/ 里的每份 BP 抽取成统一格式的 JSON。

每份存到 output/structured/<文件名>.json，最后汇总成 output/bp_table.csv。
已处理过的会跳过；出错会重试一次，仍失败的记录到 output/structured_errors.log。

用法：
  .venv/bin/python src/extract_structured.py --limit 3   # 试跑 3 份
  .venv/bin/python src/extract_structured.py             # 跑全部
"""
import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from paths import OUTPUT_DIR, ROOT, TEXT_DIR

OUT_DIR = OUTPUT_DIR / "structured"
CSV_PATH = OUTPUT_DIR / "bp_table.csv"
ERROR_LOG = OUTPUT_DIR / "structured_errors.log"

MODEL = "deepseek-chat"
NA = "未提及"
MATURITY_OPTIONS = ["实验室", "小试", "中试", "量产"]
# 固定的工艺环节清单，统一口径，方便和企业难题比对
PROCESS_STAGE_OPTIONS = [
    "AI/数字化设计", "酶设计/改造", "菌种/细胞构建", "生物催化/转化",
    "发酵/培养", "分离纯化", "制剂/成品", "分析检测", "装备/仪器",
]
STR_FIELDS = ["project_name", "company_name", "track", "core_technology",
              "process_route", "maturity", "maturity_evidence", "summary"]
LIST_FIELDS = ["target_products", "process_stages", "application_fields",
               "key_metrics", "evidence_gaps"]
FIELDS = ["project_name", "company_name", "track", "core_technology",
          "target_products", "process_stages", "process_route",
          "application_fields", "maturity", "maturity_evidence",
          "key_metrics", "evidence_gaps", "summary"]

SYSTEM_PROMPT = f"""你是生物制造领域的项目分析师，负责把参赛项目简介整理成结构化数据。

【最重要的规则】
- 只能根据用户给出的原文填写，严禁编造、推测或补充原文没有的信息。
- 原文没有提到的字段：字符串字段填"{NA}"，列表字段填["{NA}"]。

【输出格式】只输出一个 JSON 对象，包含以下字段：
- project_name（字符串）：项目名称，照抄原文
- company_name（字符串）：公司名称，照抄原文
- track（字符串）：赛道和方向，照抄原文
- core_technology（字符串）：核心技术，一句话概括
- target_products（字符串列表）：目标产物或产品
- process_stages（字符串列表）：项目涉及的工艺环节，只能从这个清单里选（可多选）：{PROCESS_STAGE_OPTIONS}。
  原文明确写到才选，不要根据技术类型推测（例如原文只写"细胞工厂"，没写发酵，就不选"发酵/培养"）。
  概念验证、临床研究、注册申报、销售供货等不属于工艺环节，不要选。
- process_route（字符串）：原文写的工艺路线，照抄原文（如"A→B→C"）
- application_fields（字符串列表）：应用的行业领域，如医药、疫苗、食品营养、美妆个护、农业、饲料、化工材料等。
  注意：只写行业，不要写客户或服务对象类型（如"疫苗企业""临床研究机构""工艺合作方"都不是领域）。
- maturity（字符串）：当前实际的产业化阶段（不是未来计划），只能从 {MATURITY_OPTIONS} 中选一个：
  原文写"已上市"或"已有销售"的，算"量产"；原文写明"报名阶段为X"的，以此为准；原文无法判断时填"{NA}"。
- maturity_evidence（字符串）：判断 maturity 所依据的原文句子，必须一字不差地照抄原文；maturity 为"{NA}"时也填"{NA}"
- key_metrics（字符串列表）：关键量化技术指标，只收带具体数字的条目，每条保留原文表述，不要改写数字或单位。
  不要收：没有数字的描述（如"无可核对数值"）、复赛得分、融资金额和估值。
- evidence_gaps（字符串列表）：数据存疑或缺失的地方，包括原文自己标注的"待核实/需补充"内容，以及缺少对照条件、口径不清或没有数值的指标
- summary（字符串）：100 字以内的总结"""


def load_client() -> OpenAI:
    load_dotenv(ROOT / ".env")
    key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        sys.exit("❌ .env 里的 DEEPSEEK_API_KEY 是空的")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=120)


def squash(s: str) -> str:
    """去掉所有空白，用于和原文比对（PDF 提取的文字中间常有断行）"""
    return "".join(s.split())


def validate(data: dict, source_text: str) -> dict:
    """检查字段是否齐全、类型是否正确；不合格就抛出异常（会触发重试）"""
    missing = [f for f in FIELDS if f not in data]
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    for f in STR_FIELDS:
        if not isinstance(data[f], str) or not data[f].strip():
            raise ValueError(f"{f} 应为非空字符串")
    for f in LIST_FIELDS:
        if not isinstance(data[f], list) or not all(isinstance(x, str) for x in data[f]):
            raise ValueError(f"{f} 应为字符串列表")

    if data["maturity"] not in MATURITY_OPTIONS + [NA]:
        raise ValueError(f"maturity 不在可选范围内：{data['maturity']}")
    if data["maturity"] != NA and squash(data["maturity_evidence"]) not in squash(source_text):
        raise ValueError(f"maturity_evidence 在原文中找不到：{data['maturity_evidence']}")
    bad_stages = [s for s in data["process_stages"] if s not in PROCESS_STAGE_OPTIONS + [NA]]
    if bad_stages:
        raise ValueError(f"process_stages 不在清单内：{bad_stages}")
    # 兜底：不带数字的指标直接去掉
    data["key_metrics"] = [m for m in data["key_metrics"] if re.search(r"\d", m)]

    for f in LIST_FIELDS:
        items = [x for x in data[f] if x != NA]
        data[f] = items or [NA]
    return {f: data[f] for f in FIELDS}


def extract_one(client: OpenAI, text: str) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"原文如下：\n\n{text}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return validate(json.loads(resp.choices[0].message.content), text)


def process(client: OpenAI, txt: Path) -> tuple[str, str | None]:
    """返回 (文件名, 错误信息)；成功时错误信息为 None"""
    text = txt.read_text(encoding="utf-8")
    last_err = None
    for _ in range(2):  # 第一次 + 重试一次
        try:
            data = extract_one(client, text)
            data["source_file"] = txt.stem
            out = OUT_DIR / f"{txt.stem}.json"
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return txt.name, None
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    return txt.name, last_err


def build_csv():
    rows = []
    for f in sorted(OUT_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        rows.append({k: "；".join(v) if isinstance(v, list) else v for k, v in data.items()})
    # utf-8-sig 让 Excel 打开时中文不乱码
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as fp:
        # application_categories 由 normalize_fields.py 补上，没跑过时留空
        writer = csv.DictWriter(fp, fieldnames=FIELDS + ["application_categories", "source_file"],
                                restval="")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="只处理前 N 份（试跑用）")
    parser.add_argument("--workers", type=int, default=5, help="同时处理几份")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    txts = sorted(TEXT_DIR.glob("*.txt"))
    if args.limit:
        txts = txts[:args.limit]
    todo = [t for t in txts if not (OUT_DIR / f"{t.stem}.json").exists()]
    print(f"共 {len(txts)} 份，已处理过 {len(txts) - len(todo)} 份（跳过），本次处理 {len(todo)} 份")

    client = load_client()
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, client, t) for t in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            name, err = fut.result()
            print(f"[{i}/{len(todo)}] {'❌' if err else '✅'} {name}")
            if err:
                errors.append((name, err))

    if errors:
        with ERROR_LOG.open("a", encoding="utf-8") as fp:
            for name, err in errors:
                fp.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{name}\t{err}\n")

    n_rows = build_csv()
    print(f"\n本次成功 {len(todo) - len(errors)} 份，失败 {len(errors)} 份")
    if errors:
        print(f"失败记录已写入 {ERROR_LOG.relative_to(ROOT)}")
    print(f"汇总表 {CSV_PATH.relative_to(ROOT)} 共 {n_rows} 行")


if __name__ == "__main__":
    main()
