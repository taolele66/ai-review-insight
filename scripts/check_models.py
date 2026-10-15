"""列出你的 key 能用的模型名，用来确认 .env 里的 QWEN_MODEL / DEEPSEEK_MODEL 填什么。
用法：python3 scripts/check_models.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openai import OpenAI  # noqa: E402

from src import config as C  # noqa: E402

for name in ("qwen", "deepseek"):
    cfg = C.MODELS[name]
    key = os.environ.get(cfg["key_env"])
    if not key:
        print(f"[{name}] .env 里没有 {cfg['key_env']}，跳过")
        continue
    try:
        ids = sorted(m.id for m in OpenAI(api_key=key, base_url=cfg["base_url"]).models.list())
        if name == "qwen":
            ids = [i for i in ids if i.startswith(("qwen3", "qwen-plus", "qwen-max", "qwen-turbo", "qwen-flash"))]
        print(f"[{name}] 当前配置 {cfg['model']}，可用：{', '.join(ids)}")
    except Exception as e:
        print(f"[{name}] 查询失败：{type(e).__name__}: {e}")
