"""第 3 步：用人工标注的 300 条评估模型输出。

用法：
    python -m src.eval.evaluate --gold annotation/annotation_300.xlsx \
        --pred data/processed/labels_deepseek_v1_eval.parquet data/processed/labels_qwen_v2_eval.parquet

指标（outputs/eval_report.md）：
- 方面识别：每个方面「是否被提到」的 P / R / F1，及 micro / macro 汇总
- 方面情绪：在人和模型都识别到的方面上，情绪是否一致（accuracy）
- 方面+情绪联合 F1：(方面, 情绪) 二元组完全一致才算对，最严格
- 痛点：多标签 micro / macro F1
- 购买动机、复购意向、整体情绪：accuracy、macro F1
- 肤质：多标签 micro F1；是否相关：accuracy
- 难例：按 hard_case 分组的方面+情绪联合 F1
- 整体情绪 accuracy 另给一个按真实星级分布加权的版本（评估集按星级等量抽样）
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from src.labeling.schema import ASPECTS, PAIN_POINTS, SENTIMENTS, SKIN_TYPES

SENT_NUM = {k: v[1] for k, v in SENTIMENTS.items()}


def load_gold(path):
    g = pd.read_excel(path, sheet_name="标注")
    g.columns = [str(c).split("\n")[0] for c in g.columns]
    rows = []
    for r in g.itertuples(index=False):
        r = r._asdict()
        if pd.isna(r.get("overall_sentiment")):
            continue  # 没标完的跳过
        row = {"review_id": r["review_id"], "hard_case": r.get("hard_case") if pd.notna(r.get("hard_case")) else "普通"}
        for a in ASPECTS:
            v = r.get(f"aspect_{a}")
            row[f"aspect_{a}"] = SENT_NUM.get(str(v).strip()) if pd.notna(v) else None
        pains = {p.strip() for p in str(r.get("pain_points") or "").split(",") if p.strip() and p.strip() != "nan"}
        unknown = pains - set(PAIN_POINTS)
        if unknown:
            print(f"  警告 {r['review_id']} 有未知痛点代码 {unknown}")
        for p in PAIN_POINTS:
            row[f"pain_{p}"] = int(p in pains)
        skins = {t.strip() for t in str(r.get("skin_type") or "").split(",") if t.strip() and t.strip() != "nan"}
        for t in SKIN_TYPES:
            row[f"skin_{t}"] = int(t in skins)
        row["purchase_motive"] = r.get("purchase_motive") if pd.notna(r.get("purchase_motive")) else "none"
        row["repurchase_intent"] = r.get("repurchase_intent") if pd.notna(r.get("repurchase_intent")) else "none"
        rel = r.get("is_relevant")
        row["is_relevant"] = not (str(rel).strip().upper() == "FALSE") if pd.notna(rel) else True
        row["overall_sentiment"] = r["overall_sentiment"]
        rows.append(row)
    return pd.DataFrame(rows)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f = 2 * p * r / (p + r) if p + r and not np.isnan(p + r) else np.nan
    return p, r, f


def macro_f1(y_true, y_pred):
    fs = []
    for c in set(y_true) | set(y_pred):
        tp = sum((t == c) & (p == c) for t, p in zip(y_true, y_pred))
        fp = sum((t != c) & (p == c) for t, p in zip(y_true, y_pred))
        fn = sum((t == c) & (p != c) for t, p in zip(y_true, y_pred))
        fs.append(prf(tp, fp, fn)[2])
    return np.nanmean(fs)


def evaluate(gold, pred, star_weights=None):
    m = gold.merge(pred, on="review_id", suffixes=("_g", "_p"))
    res = {"n": len(m), "n_pred_failed": int((~m["label_ok"]).sum())}

    per_aspect, TP, FP, FN, jTP, jFP, jFN, agree, both = [], 0, 0, 0, 0, 0, 0, 0, 0
    for a in ASPECTS:
        g, p = m[f"aspect_{a}_g"], m[f"aspect_{a}_p"]
        gm, pm = g.notna(), p.notna()
        tp, fp, fn = int((gm & pm).sum()), int((~gm & pm).sum()), int((gm & ~pm).sum())
        TP += tp; FP += fp; FN += fn
        same = (gm & pm) & (g == p)
        agree += int(same.sum()); both += tp
        jtp = int(same.sum())
        jTP += jtp; jFP += int(pm.sum()) - jtp; jFN += int(gm.sum()) - jtp
        P, R, F = prf(tp, fp, fn)
        per_aspect.append({"aspect": a, "中文": ASPECTS[a][0], "support": int(gm.sum()), "P": P, "R": R, "F1": F,
                           "情绪一致率": same.sum() / tp if tp else np.nan})
    per_aspect = pd.DataFrame(per_aspect)
    res["aspect_micro_f1"] = prf(TP, FP, FN)[2]
    res["aspect_macro_f1"] = per_aspect.loc[per_aspect.support > 0, "F1"].mean()
    res["aspect_sentiment_acc"] = agree / both if both else np.nan
    res["aspect_sentiment_joint_f1"] = prf(jTP, jFP, jFN)[2]

    pTP = pFP = pFN = 0
    pf = []
    for pp in PAIN_POINTS:
        g, p = m[f"pain_{pp}_g"].astype(int), m[f"pain_{pp}_p"].fillna(0).astype(int)
        tp, fp, fn = int((g & p).sum()), int((~g.astype(bool) & p.astype(bool)).sum()), int((g.astype(bool) & ~p.astype(bool)).sum())
        pTP += tp; pFP += fp; pFN += fn
        if g.sum():
            pf.append(prf(tp, fp, fn)[2])
    res["pain_micro_f1"] = prf(pTP, pFP, pFN)[2]
    res["pain_macro_f1"] = np.nanmean(pf) if pf else np.nan

    sTP = sFP = sFN = 0
    for t in SKIN_TYPES:
        g, p = m[f"skin_{t}_g"].astype(bool), m[f"skin_{t}_p"].fillna(0).astype(bool)
        sTP += int((g & p).sum()); sFP += int((~g & p).sum()); sFN += int((g & ~p).sum())
    res["skin_type_micro_f1"] = prf(sTP, sFP, sFN)[2]

    rel_g, rel_p = m["is_relevant_g"].astype(bool), m["is_relevant_p"].fillna(True).astype(bool)
    res["is_relevant_acc"] = float((rel_g == rel_p).mean())

    for col in ("purchase_motive", "repurchase_intent", "overall_sentiment"):
        g, p = m[f"{col}_g"].tolist(), m[f"{col}_p"].fillna("FAILED").tolist()
        res[f"{col}_acc"] = np.mean([a == b for a, b in zip(g, p)])
        res[f"{col}_macro_f1"] = macro_f1(g, p)

    if star_weights is not None and "rating" in m:
        ok = (m["overall_sentiment_g"] == m["overall_sentiment_p"]).groupby(m["rating"]).mean()
        res["overall_sentiment_acc_weighted"] = float((ok * star_weights.reindex(ok.index)).sum())

    hard = []
    for h, sub in m.groupby("hard_case"):
        t = f = n_ = 0
        for a in ASPECTS:
            g, p = sub[f"aspect_{a}_g"], sub[f"aspect_{a}_p"]
            same = int((g.notna() & p.notna() & (g == p)).sum())
            t += same; f += int(p.notna().sum()) - same; n_ += int(g.notna().sum()) - same
        hard.append({"类型": h, "n": len(sub), "方面+情绪 F1": prf(t, f, n_)[2]})
    return res, per_aspect, pd.DataFrame(hard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(C.ROOT / "annotation" / "annotation_300.xlsx"))
    ap.add_argument("--pred", nargs="+", required=True)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    sample = pd.read_parquet(C.PROC_DIR / "reviews_clean.parquet", columns=["review_id", "rating"])
    gold = gold.merge(sample, on="review_id", how="left")
    star_w = sample["rating"].value_counts(normalize=True)
    print(f"人工标注完成 {len(gold)} 条")

    stats = pd.read_csv(C.OUT_DIR / "run_stats.csv") if (C.OUT_DIR / "run_stats.csv").exists() else None
    lines = ["# 评估结果", "", f"人工标注 {len(gold)} 条（按星级分层抽样）。", ""]
    summary = []
    for path in args.pred:
        name = Path(path).stem.replace("labels_", "")
        res, per_aspect, hard = evaluate(gold, pd.read_parquet(path), star_w)
        if stats is not None:
            prov, prompt = name.split("_")[:2]
            s = stats[(stats.provider == prov) & (stats.prompt == prompt) & stats.cost_per_1k_cny.notna()]
            if len(s):
                res["cost_per_1k_cny"] = s.iloc[-1].cost_per_1k_cny
                res["sec_per_1k"] = s.iloc[-1].sec_per_1k
        summary.append({"方案": name, **res})
        lines += [f"## {name}", "", "### 各方面", "", per_aspect.round(3).to_markdown(index=False),
                  "", "### 难例", "", hard.round(3).to_markdown(index=False), ""]
    summ = pd.DataFrame(summary).set_index("方案").T.round(3)
    lines[4:4] = ["## 方案对比", "", summ.to_markdown(), ""]
    C.OUT_DIR.mkdir(exist_ok=True)
    (C.OUT_DIR / "eval_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(summ.to_string())
    print("详细结果见 outputs/eval_report.md")


if __name__ == "__main__":
    main()
