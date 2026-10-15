"""第 1 步：下载 + 清洗 Amazon Reviews 2023 护肤评论（先跑 src.prefilter 筛出护肤子品类）。

用法：
    python -m src.data_prep --download      # 下载原始文件到 data/raw/
    python -m src.data_prep                 # 清洗 + 抽样，输出 data/processed/reviews_clean.parquet

清洗规则（每一步的剩余条数都会打印并写进 outputs/cleaning_log.json，报告里直接引用）：
    1. 去掉空文本
    2. 限定时间段 START_DATE ~ END_DATE
    3. 去重：同一用户对同一商品的同样文本只留一条
    4. 去掉过短评论（< MIN_CHARS 字符），超长评论截断
    5. 关联商品元数据（品牌、价格、标题），随机抽 SAMPLE_SIZE 条作为分析主样本
"""
import argparse
import gzip
import json
import re
import urllib.request

import numpy as np
import pandas as pd

from src import config as C

RAW_REVIEWS = C.RAW_DIR / f"{C.CATEGORY}.jsonl.gz"
RAW_META = C.RAW_DIR / f"meta_{C.CATEGORY}.jsonl.gz"


def download():
    C.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for url, dest in [(C.REVIEW_URL, C.RAW_DIR / f"{C.SOURCE_CATEGORY}.jsonl.gz"),
                      (C.META_URL, C.RAW_DIR / f"meta_{C.SOURCE_CATEGORY}.jsonl.gz")]:
        if dest.exists():
            print(f"已存在，跳过 {dest.name}")
            continue
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, dest)


def _open(path):
    # 兼容 .jsonl.gz 和 HuggingFace 上的未压缩 .jsonl
    if path.exists():
        return gzip.open(path, "rt", encoding="utf-8")
    plain = path.with_suffix("")
    return open(plain, "rt", encoding="utf-8")


def load_reviews():
    rows = []
    with _open(RAW_REVIEWS) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "rating": r["rating"],
                "title": r.get("title", ""),
                "text": r.get("text", ""),
                "asin": r["asin"],
                "parent_asin": r["parent_asin"],
                "user_id": r["user_id"],
                "timestamp": r["timestamp"],
                "helpful_vote": r.get("helpful_vote", 0),
                "verified_purchase": r.get("verified_purchase", False),
                "has_image": bool(r.get("images")),
            })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def _brand(m):
    details = m.get("details") or {}
    return (details.get("Brand") or m.get("store") or "").strip() or None


def _price(p):
    if p is None:
        return np.nan
    try:
        return float(str(p).replace("$", "").replace(",", ""))
    except ValueError:
        return np.nan


def load_meta():
    rows = []
    with _open(RAW_META) as f:
        for line in f:
            m = json.loads(line)
            cats = m.get("categories") or []
            rows.append({
                "parent_asin": m["parent_asin"],
                "product_title": m.get("title", ""),
                "brand": _brand(m),
                "price": _price(m.get("price")),
                "item_avg_rating": m.get("average_rating"),
                "item_rating_number": m.get("rating_number"),
                # categories 形如 ["Beauty & Personal Care", "Skin Care", "Face", "Creams & Moisturizers", ...]
                "sub_category": cats[2] if len(cats) > 2 else None,
                "leaf_category": cats[-1] if len(cats) > 2 else None,
            })
    return pd.DataFrame(rows).drop_duplicates("parent_asin")


def clean_text(s):
    s = re.sub(r"<br\s*/?>", " ", s or "", flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean(df, meta):
    log = {}
    pre = C.OUT_DIR / "prefilter_log.json"
    if pre.exists():
        p = json.loads(pre.read_text())
        log["source_all"] = p["reviews_all"]
        log["skincare"] = p["reviews_skincare"]
    log["raw"] = len(df)

    df["text"] = df["text"].map(clean_text)
    df["title"] = df["title"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    log["non_empty"] = len(df)

    df = df[(df["date"] >= C.START_DATE) & (df["date"] <= C.END_DATE + " 23:59:59")]
    log["in_date_range"] = len(df)

    df = df.drop_duplicates(["user_id", "parent_asin", "text"])
    log["dedup"] = len(df)

    df = df[df["text"].str.len() >= C.MIN_CHARS].copy()
    df["text"] = df["text"].str.slice(0, C.MAX_CHARS)
    log["min_length"] = len(df)

    df = df.merge(meta, on="parent_asin", how="left")
    # 价格段：在商品层面（不是评论层面）按价格三等分，避免爆款商品左右分档
    items = df.drop_duplicates("parent_asin")[["parent_asin", "price"]].dropna()
    items["price_band"] = pd.qcut(items["price"].rank(method="first"), 3,
                                  labels=["低价", "中价", "高价"]).astype("object")
    df = df.merge(items[["parent_asin", "price_band"]], on="parent_asin", how="left")
    df["price_band"] = df["price_band"].fillna("未知")
    df["year_quarter"] = df["date"].dt.to_period("Q").astype(str)

    full = df
    if len(df) > C.SAMPLE_SIZE:
        df = df.sample(C.SAMPLE_SIZE, random_state=C.SEED)
    log["sample"] = len(df)

    df = df.sort_values("date").reset_index(drop=True)
    df.insert(0, "review_id", [f"r_{i:06d}" for i in range(len(df))])
    return df, full, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    if args.download:
        download()
        return

    C.PROC_DIR.mkdir(parents=True, exist_ok=True)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    reviews, meta = load_reviews(), load_meta()
    sample, full, log = clean(reviews, meta)
    for k, v in log.items():
        print(f"{k:>14}: {v:,}")

    sample.to_parquet(C.PROC_DIR / "reviews_clean.parquet", index=False)
    # 全量清洗后的表也存一份，趋势分析可以用全量（不需要 LLM 标签的指标）
    full.drop(columns=["timestamp"]).to_parquet(C.PROC_DIR / "reviews_clean_full.parquet", index=False)
    with open(C.OUT_DIR / "cleaning_log.json", "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(sample):,} 条到 data/processed/reviews_clean.parquet")


if __name__ == "__main__":
    main()
