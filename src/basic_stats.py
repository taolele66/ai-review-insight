"""第 1 步：基础统计，输出 outputs/basic_stats.md 和图表，直接作为报告「数据背景」一节。

用法：python -m src.basic_stats
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C


def pct(s):
    return (s / s.sum() * 100).round(1)


def main():
    df = pd.read_parquet(C.PROC_DIR / "reviews_clean.parquet")
    full = pd.read_parquet(C.PROC_DIR / "reviews_clean_full.parquet")
    log = json.loads((C.OUT_DIR / "cleaning_log.json").read_text())
    fig_dir = C.OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lines = ["# 数据背景：Amazon Reviews 2023 · Beauty & Personal Care · 护肤", ""]
    lines += ["## 清洗漏斗", "", "| 步骤 | 剩余评论数 |", "|---|---|"]
    names = {"source_all": f"{C.SOURCE_CATEGORY} 全部评论", "skincare": "护肤子品类",
             "raw": f"护肤 · {C.START_DATE[:4]} 年以后", "non_empty": "去空文本", "in_date_range": f"限定 {C.START_DATE[:4]}–{C.END_DATE[:4]}",
             "dedup": "去重", "min_length": f"去掉 <{C.MIN_CHARS} 字符", "sample": "随机抽样（分析主样本）"}
    lines += [f"| {names[k]} | {v:,} |" for k, v in log.items()]

    lines += ["", "## 分析主样本概况", ""]
    lines += [f"- 评论数：{len(df):,}；商品数：{df['parent_asin'].nunique():,}；品牌数：{df['brand'].nunique():,}；用户数：{df['user_id'].nunique():,}",
              f"- 时间：{df['date'].min():%Y-%m-%d} 至 {df['date'].max():%Y-%m-%d}",
              f"- 平均评分 {df['rating'].mean():.2f}；差评（≤2 星）占比 {(df['rating'] <= 2).mean():.1%}",
              f"- 认证购买占比 {df['verified_purchase'].mean():.1%}；带图评论占比 {df['has_image'].mean():.1%}",
              f"- 评论长度中位数 {df['text'].str.len().median():.0f} 字符；有品牌信息 {df['brand'].notna().mean():.1%}；有价格信息 {df['price'].notna().mean():.1%}"]

    rd = df["rating"].value_counts().sort_index()
    lines += ["", "## 评分分布", "", "| 星级 | 条数 | 占比 |", "|---|---|---|"]
    lines += [f"| {int(k)} | {v:,} | {p}% |" for (k, v), p in zip(rd.items(), pct(rd))]

    lines += ["", "## 价格段", "", "| 价格段 | 条数 | 平均评分 |", "|---|---|---|"]
    for band, g in df.groupby("price_band"):
        lines.append(f"| {band} | {len(g):,} | {g['rating'].mean():.2f} |")

    top = (df.groupby("brand").agg(n=("rating", "size"), avg=("rating", "mean"),
                                   neg=("rating", lambda s: (s <= 2).mean()))
           .sort_values("n", ascending=False).head(15))
    lines += ["", "## 评论量 Top 15 品牌", "", "| 品牌 | 评论数 | 平均评分 | 差评率 |", "|---|---|---|---|"]
    lines += [f"| {b} | {int(r.n):,} | {r.avg:.2f} | {r.neg:.1%} |" for b, r in top.iterrows()]

    for col, title in [("sub_category", "二级品类"), ("leaf_category", "末级品类 Top 15")]:
        if df[col].notna().mean() > 0.2:
            g = (df.groupby(col).agg(n=("rating", "size"), avg=("rating", "mean"))
                 .sort_values("n", ascending=False).head(15))
            lines += ["", f"## {title}", "", "| 品类 | 条数 | 平均评分 |", "|---|---|---|"]
            lines += [f"| {k} | {int(r.n):,} | {r.avg:.2f} |" for k, r in g.iterrows()]

    # 图：评分分布、季度评论量与平均评分（全量清洗后数据，趋势更稳）
    fig, ax = plt.subplots(figsize=(5, 3))
    rd.plot.bar(ax=ax, color="#4C72B0")
    ax.set_xlabel("rating"); ax.set_ylabel("reviews"); ax.set_title("Rating distribution (sample)")
    fig.tight_layout(); fig.savefig(fig_dir / "rating_dist.png", dpi=150); plt.close(fig)

    q = full.groupby("year_quarter").agg(n=("rating", "size"), avg=("rating", "mean"))
    fig, ax1 = plt.subplots(figsize=(8, 3))
    ax1.bar(q.index, q["n"], color="#C5D3E8"); ax1.set_ylabel("reviews")
    ax2 = ax1.twinx(); ax2.plot(q.index, q["avg"], color="#C44E52", marker="o"); ax2.set_ylabel("avg rating")
    ax1.tick_params(axis="x", rotation=60); ax1.set_title("Quarterly volume and avg rating (full cleaned data)")
    fig.tight_layout(); fig.savefig(fig_dir / "quarterly_trend.png", dpi=150); plt.close(fig)
    lines += ["", "![评分分布](figures/rating_dist.png)", "", "![季度趋势](figures/quarterly_trend.png)", ""]

    (C.OUT_DIR / "basic_stats.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
