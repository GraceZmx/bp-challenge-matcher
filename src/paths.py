"""数据集路径集中放在这里。

默认使用真实数据（data/、output/，不公开）；
设置环境变量 BPM_DATASET=demo 时使用 demo_data/ 里的合成数据（可公开）：
  BPM_DATASET=demo .venv/bin/python src/extract_structured.py
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = os.getenv("BPM_DATASET", "real")

if DATASET == "real":
    TEXT_DIR = ROOT / "output" / "texts"          # 由 extract_pdfs.py 从 PDF 转出
    CHALLENGE_XLSX = ROOT / "data" / "揭榜挂帅企业赛题汇总.xlsx"
    OUTPUT_DIR = ROOT / "output"
    CHALLENGE_OVERRIDES = ROOT / "config" / "challenge_overrides.json"
elif DATASET == "demo":
    TEXT_DIR = ROOT / "demo_data" / "bps"         # 合成 BP 直接就是文本
    CHALLENGE_XLSX = ROOT / "demo_data" / "challenges.xlsx"
    OUTPUT_DIR = ROOT / "demo_data" / "output"
    CHALLENGE_OVERRIDES = None                    # 人工修正只针对真实数据
else:
    raise SystemExit(f"BPM_DATASET 只能是 real 或 demo：{DATASET}")
