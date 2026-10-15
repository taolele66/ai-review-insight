"""全局配置：路径、数据范围、模型列表与计价。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # 先读 .env，下面的 QWEN_MODEL / DEEPSEEK_MODEL 覆盖才生效
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "outputs"
CACHE_DIR = ROOT / "cache"

# Amazon Reviews 2023 (McAuley Lab)，Beauty_and_Personal_Care 类目里的护肤（Skin Care）子品类
# 原始大文件先用 src/prefilter.py 筛成 Skincare.jsonl.gz，后面的脚本都读筛好的文件
SOURCE_CATEGORY = "Beauty_and_Personal_Care"
CATEGORY = "Skincare"
REVIEW_URL = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{SOURCE_CATEGORY}.jsonl.gz"
META_URL = f"https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/meta_{SOURCE_CATEGORY}.jsonl.gz"

# 清洗参数
START_DATE = "2019-01-01"   # 只保留近几年，口碑趋势更有意义
END_DATE = "2023-12-31"
MIN_CHARS = 20              # 太短的评论（"Great!"）几乎没有方面信息
MAX_CHARS = 2000            # 截断超长评论，控制 token 成本
SAMPLE_SIZE = 30000         # 分析主样本量（大纲建议 2–5 万）
SEED = 42

# 模型：都走 OpenAI 兼容接口。价格单位：元 / 百万 token，以官网实时价格为准，跑之前核对一遍
MODELS = {
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "key_env": "DEEPSEEK_API_KEY",
        "price_in": 2.0, "price_out": 8.0,
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # 千问：默认 qwen-plus（百炼通用模型名）；想指定千问 3 的具体版本，用 scripts/check_models.py 查账号里可用的名字，再在 .env 里设 QWEN_MODEL
        "model": os.environ.get("QWEN_MODEL", "qwen-plus"),
        "key_env": "DASHSCOPE_API_KEY",
        "price_in": 0.8, "price_out": 2.0,
    },
    # 不需要 key 的假模型，用来测试整条 pipeline
    "mock": {"base_url": None, "model": "mock", "key_env": None, "price_in": 0, "price_out": 0},
}
