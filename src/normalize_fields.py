"""按 config/application_field_map.json，把每份 BP 的应用领域归并到固定类别。

结果写入每个 json 的 application_categories 字段（原始的 application_fields 保留不动），
然后重新生成 output/bp_table.csv。不调用大模型。

用法：.venv/bin/python src/normalize_fields.py
"""
import json
import sys
from pathlib import Path

from extract_structured import NA, OUT_DIR, ROOT, build_csv

MAP_PATH = ROOT / "config" / "application_field_map.json"


def main():
    category_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    term_to_cat = {term: cat for cat, terms in category_map.items() for term in terms}
    order = list(category_map)

    unmapped = {}
    files = sorted(OUT_DIR.glob("*.json"))
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        cats = set()
        for term in data["application_fields"]:
            if term == NA:
                continue
            if term in term_to_cat:
                cats.add(term_to_cat[term])
            else:
                unmapped.setdefault(term, []).append(f.stem)
        data["application_categories"] = sorted(cats, key=order.index) or [NA]
        f.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    if unmapped:
        print("⚠️  以下写法不在对照表里，请补到 config/application_field_map.json 后重跑：")
        for term, names in unmapped.items():
            print(f"   - {term}（{len(names)} 份，如：{names[0]}）")
    print(f"已归并 {len(files)} 份，汇总表共 {build_csv()} 行")
    sys.exit(1 if unmapped else 0)


if __name__ == "__main__":
    main()
