"""没有人工标注时，先比较几组模型标注之间的一致性、标签分布和成本。

一致性高不代表准，但两个模型分歧大的标签，通常就是定义不清、需要改规则的地方。

用法：python -m src.eval.compare_runs data/processed/labels_*_eval.parquet
输出：outputs/model_agreement.md
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.labeling.schema import ASPECTS, PAIN_POINTS, SKIN_TYPES


def aspect_sets(d, with_sentiment):
    cols = [f"aspect_{a}" for a in ASPECTS]
    out = {}
    for rid, row in d[cols].iterrows():
        out[rid] = {(c[7:], row[c]) if with_sentiment else c[7:] for c in cols if pd.notna(row[c])}
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    args = ap.parse_args()

    runs = {Path(p).stem.replace("labels_", "").replace("_eval", ""): pd.read_parquet(p).set_index("review_id")
            for p in args.preds}
    ids = sorted(set.intersection(*[set(d.index[d["label_ok"]]) for d in runs.values()]))
    rating = pd.read_parquet(C.PROC_DIR / "reviews_clean.parquet").set_index("review_id").loc[ids, "rating"]
    star_sent = np.where(rating >= 4, "pos", np.where(rating <= 2, "neg", "neu"))

    lines = ["# 模型标注对比（无人工标注版）", "",
             f"共 {len(ids)} 条评论在所有方案里都标注成功。一致性高不等于准确，准确率要等人工标注后用 evaluate.py 算。", ""]

    rows = []
    for n, d in runs.items():
        d = d.loc[ids]
        rows.append({
            "方案": n, "失败条数": int((~runs[n]["label_ok"]).sum()),
            "每条方面数": d[[f"aspect_{a}" for a in ASPECTS]].notna().sum(axis=1).mean(),
            "每条痛点数": d[[f"pain_{p}" for p in PAIN_POINTS]].sum(axis=1).mean(),
            "有购买动机": (d["purchase_motive"] != "none").mean(),
            "复购 yes": (d["repurchase_intent"] == "yes").mean(),
            "复购 no": (d["repurchase_intent"] == "no").mean(),
            "提到肤质": d[[f"skin_{t}" for t in SKIN_TYPES]].sum(axis=1).gt(0).mean(),
            "判为无关": int((d["is_relevant"] == False).sum()),  # noqa: E712
            "整体情绪与星级一致": (d["overall_sentiment"].values == star_sent).mean(),
        })
    lines += ["## 标签分布", "", pd.DataFrame(rows).round(2).to_markdown(index=False), ""]

    stats_path = C.OUT_DIR / "run_stats.csv"
    if stats_path.exists():
        s = pd.read_csv(stats_path)
        s = s[s["n_api_calls"] > 0].drop_duplicates(["provider", "prompt", "tag"], keep="last")
        lines += ["## 成本与速度（按 config.py 里的单价估算，以账单为准）", "",
                  s[["provider", "model", "prompt", "n_api_calls", "n_failed", "cost_cny",
                     "cost_per_1k_cny", "sec_per_1k"]].to_markdown(index=False), ""]

    pair = []
    sets = {n: (aspect_sets(d.loc[ids], False), aspect_sets(d.loc[ids], True)) for n, d in runs.items()}
    for x, y in itertools.combinations(runs, 2):
        pair.append({"对比": f"{x} vs {y}",
                     "方面重合度": np.mean([jaccard(sets[x][0][i], sets[y][0][i]) for i in ids]),
                     "方面+情绪重合度": np.mean([jaccard(sets[x][1][i], sets[y][1][i]) for i in ids]),
                     "整体情绪一致": (runs[x].loc[ids, "overall_sentiment"] == runs[y].loc[ids, "overall_sentiment"]).mean(),
                     "复购意向一致": (runs[x].loc[ids, "repurchase_intent"] == runs[y].loc[ids, "repurchase_intent"]).mean(),
                     "购买动机一致": (runs[x].loc[ids, "purchase_motive"] == runs[y].loc[ids, "purchase_motive"]).mean()})
    lines += ["## 两两一致性（Jaccard / 一致率）", "", pd.DataFrame(pair).round(2).to_markdown(index=False), ""]

    freq = pd.DataFrame({n: d.loc[ids][[f"aspect_{a}" for a in ASPECTS]].notna().mean() for n, d in runs.items()})
    freq.index = [ASPECTS[c[7:]][0] for c in freq.index]
    lines += ["## 各方面被提到的比例", "", freq.round(2).to_markdown(), ""]

    out = C.OUT_DIR / "model_agreement.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
