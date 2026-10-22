"""把 data/ 里的所有 PDF 转成文字，每份存成一个 txt 到 output/texts/。

用法：.venv/bin/python src/extract_pdfs.py
"""
from pathlib import Path

import fitz  # pymupdf 的导入名

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output" / "texts"

# 平均每页少于这么多字，就认为基本没提取到文字（很可能是扫描件/图片）
MIN_CHARS_PER_PAGE = 20


def extract_one(pdf_path: Path) -> tuple[str, int]:
    """返回 (全文文字, 页数)"""
    with fitz.open(pdf_path) as doc:
        pages = [page.get_text() for page in doc]
        return "\n\n".join(pages), len(pages)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() == ".pdf")

    ok, no_text, failed = [], [], []
    for pdf in pdfs:
        try:
            text, n_pages = extract_one(pdf)
        except Exception as e:
            failed.append((pdf.name, str(e)))
            continue

        n_chars = len("".join(text.split()))  # 去掉空白后的字数
        (OUT_DIR / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")

        if n_pages == 0 or n_chars / n_pages < MIN_CHARS_PER_PAGE:
            no_text.append((pdf.name, n_pages, n_chars))
        else:
            ok.append((pdf.name, n_pages, n_chars))

    print(f"共 {len(pdfs)} 份 PDF")
    print(f"✅ 成功提取文字：{len(ok)} 份")
    print(f"⚠️  几乎没有文字（可能是扫描件）：{len(no_text)} 份")
    for name, n_pages, n_chars in no_text:
        print(f"   - {name}（{n_pages} 页，{n_chars} 字）")
    print(f"❌ 打不开/出错：{len(failed)} 份")
    for name, err in failed:
        print(f"   - {name}：{err}")


if __name__ == "__main__":
    main()
