"""测试 .env 里的 DEEPSEEK_API_KEY 能不能连上 DeepSeek（不会打印密钥）。

用法：.venv/bin/python src/test_deepseek.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
if not key:
    sys.exit("❌ .env 里的 DEEPSEEK_API_KEY 是空的，请检查是否填写并保存")

client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
try:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "请只回复：连接成功"}],
        max_tokens=10,
    )
except Exception as e:
    # 只打印错误类型和状态码，避免把请求细节打出来
    status = getattr(e, "status_code", "")
    sys.exit(f"❌ 连接失败：{type(e).__name__} {status}")

print("✅ 连接成功！DeepSeek 回复：", resp.choices[0].message.content)
