"""选品类用：只读 meta 文件，按子品类统计商品数、价格覆盖率、品牌数、评论量（meta 里的 rating_number 之和）。

只用标准库，可以直接在大文件上流式跑。
用法：python3 -m src.profile_meta data/raw/meta_Beauty_and_Personal_Care.jsonl.gz [--depth 3]
评论量按商品加权的价格覆盖率（ratings_price_coverage）更重要：热门商品有价格，冷门缺价格影响不大。
输出：outputs/profile_<文件名>.csv，并打印评论量 Top 30 的子品类
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

from src import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--depth", type=int, default=3, help="品类路径取几级，例如 3 = 大类 > 二级 > 三级")
    args = ap.parse_args()

    stats = defaultdict(lambda: {"items": 0, "priced": 0, "reviews": 0, "reviews_priced": 0, "brands": set()})
    n = 0
    with gzip.open(args.path, "rt", encoding="utf-8") as f:
        for line in f:
            n += 1
            m = json.loads(line)
            cats = m.get("categories") or []
            if len(cats) < 2:
                key = "(无品类)"
            else:
                key = " > ".join(cats[1:args.depth])
            s = stats[key]
            s["items"] += 1
            p = m.get("price")
            r = m.get("rating_number") or 0
            s["reviews"] += r
            if p not in (None, "", "None"):
                s["priced"] += 1
                s["reviews_priced"] += r
            brand = ((m.get("details") or {}).get("Brand") or m.get("store") or "").strip()
            if brand:
                s["brands"].add(brand)

    rows = []
    for k, s in stats.items():
        rows.append({"category": k, "items": s["items"], "price_coverage": round(s["priced"] / s["items"], 3),
                     "brands": len(s["brands"]), "total_ratings": s["reviews"],
                     "ratings_price_coverage": round(s["reviews_priced"] / s["reviews"], 3) if s["reviews"] else 0})
    rows.sort(key=lambda r: -r["total_ratings"])
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = C.OUT_DIR / f"profile_{Path(args.path).name.split('.')[0]}_d{args.depth}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    priced = sum(s["priced"] for s in stats.values())
    print(f"共 {n:,} 个商品，有价格 {priced / n:.1%}，{len(rows)} 个子品类；已保存 {out}")
    for r in rows[:30]:
        print(f"{r['category'][:60]:<60} 商品 {r['items']:>7,}  价格覆盖 {r['price_coverage']:.0%}  "
              f"品牌 {r['brands']:>6,}  评论量 {r['total_ratings']:>10,}  评论按价格覆盖 {r['ratings_price_coverage']:.0%}")


if __name__ == "__main__":
    main()
