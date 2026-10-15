"""批量标注，一键重跑（命中缓存的条目不再调用 API）。

用法：
    python -m src.labeling.run_labeling --provider mock --prompt v1 --limit 200          # 无 key 跑通流程
    python -m src.labeling.run_labeling --provider deepseek --prompt v2 --ids data/processed/eval_ids.txt
    python -m src.labeling.run_labeling --provider qwen --prompt v2                      # 全量
    python -m src.labeling.run_labeling --provider qwen --prompt v2 --make-batch-file    # 生成百炼 Batch 的 JSONL
    python -m src.labeling.run_labeling --provider qwen --prompt v2 --import-batch <百炼结果.jsonl>

输出：
    data/processed/labels_<provider>_<prompt>.parquet   一条评论一行，含原始 JSON 和展开后的宽表列
    outputs/run_stats.csv                               每次运行的条数、失败数、耗时、token、费用（含每千条）
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd

from src import config as C
from src.labeling.llm_client import Labeler
from src.labeling.prompts import build_messages
from src.labeling.schema import ASPECTS, PAIN_POINTS, SENTIMENTS, SKIN_TYPES


def to_wide(results):
    """JSON -> 宽表：aspect_<x> = +1/0/-1（未提及为空），pain_<x> / skin_<x> = 0/1。"""
    rows = []
    for r in results:
        row = {"review_id": r["review_id"], "label_ok": r["labels"] is not None,
               "labels_json": json.dumps(r["labels"], ensure_ascii=False) if r["labels"] else None}
        lab = r["labels"] or {}
        for a in ASPECTS:
            row[f"aspect_{a}"] = None
        for a in lab.get("aspects", []):
            row[f"aspect_{a['aspect']}"] = SENTIMENTS[a["sentiment"]][1]
        for p in PAIN_POINTS:
            row[f"pain_{p}"] = int(p in lab.get("pain_points", [])) if lab else None
        for t in SKIN_TYPES:
            row[f"skin_{t}"] = int(t in lab.get("skin_type", [])) if lab else None
        row["purchase_motive"] = lab.get("purchase_motive")
        row["repurchase_intent"] = lab.get("repurchase_intent")
        row["is_relevant"] = lab.get("is_relevant")
        row["overall_sentiment"] = lab.get("overall_sentiment")
        rows.append(row)
    return pd.DataFrame(rows)


def make_batch_file(df, provider, prompt):
    """阿里云百炼批量推理输入文件（OpenAI Batch 格式）。结果回来后用 --import-batch 合并。"""
    cfg = C.MODELS[provider]
    path = C.PROC_DIR / f"batch_input_{provider}_{prompt}.jsonl"
    with open(path, "w") as f:
        for r in df.itertuples():
            body = {"model": cfg["model"], "temperature": 0, "response_format": {"type": "json_object"},
                    "messages": build_messages(prompt, r.title, r.text)}
            f.write(json.dumps({"custom_id": r.review_id, "method": "POST",
                                "url": "/v1/chat/completions", "body": body}, ensure_ascii=False) + "\n")
    print(f"已生成 {path}（{len(df):,} 条）")


def import_batch(path):
    """解析百炼 Batch 输出文件，返回和 Labeler.label 相同结构的结果列表。"""
    from src.labeling.llm_client import parse_json
    from src.labeling.schema import validate
    results = []
    for line in open(path):
        r = json.loads(line)
        rid = r["custom_id"]
        try:
            body = r["response"]["body"]
            labels, errs = validate(parse_json(body["choices"][0]["message"]["content"]))
            u = body.get("usage", {})
            results.append({"review_id": rid, "labels": None if errs else labels, "cached": False,
                            "tokens_in": u.get("prompt_tokens", 0), "tokens_out": u.get("completion_tokens", 0)})
        except Exception:
            results.append({"review_id": rid, "labels": None, "cached": False, "tokens_in": 0, "tokens_out": 0})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=list(C.MODELS))
    ap.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    ap.add_argument("--input", default=str(C.PROC_DIR / "reviews_clean.parquet"))
    ap.add_argument("--ids", help="只标注这些 review_id（一行一个），例如评估集")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="输出文件名后缀，例如 eval")
    ap.add_argument("--make-batch-file", action="store_true")
    ap.add_argument("--import-batch", help="百炼 Batch 结果文件路径")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.import_batch:
        results = import_batch(args.import_batch)
        out = C.PROC_DIR / f"labels_{args.provider}_{args.prompt}{suffix}.parquet"
        to_wide(results).to_parquet(out, index=False)
        bad = sum(r["labels"] is None for r in results)
        print(f"已保存 {out}（{len(results):,} 条，失败 {bad} 条，失败条目可用普通接口重跑）")
        return

    df = pd.read_parquet(args.input)
    if args.ids:
        ids = set(open(args.ids).read().split())
        df = df[df["review_id"].isin(ids)]
    if args.limit:
        df = df.head(args.limit)

    if args.make_batch_file:
        make_batch_file(df, args.provider, args.prompt)
        return

    labeler = Labeler(args.provider, args.prompt)
    labeler.preflight()
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(labeler.label, r.review_id, r.title, r.text) for r in df.itertuples()]
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 200 == 0 or i == len(futs):
                print(f"  {i:,}/{len(futs):,}")
    wall = time.time() - t0

    wide = to_wide(results)
    out = C.PROC_DIR / f"labels_{args.provider}_{args.prompt}{suffix}.parquet"
    wide.to_parquet(out, index=False)

    new = [r for r in results if not r["cached"]]
    tin = sum(r["tokens_in"] for r in new)
    tout = sum(r["tokens_out"] for r in new)
    n_fail = sum(r["labels"] is None for r in results)
    cost = labeler.cost(tin, tout)
    stats = {
        "time": datetime.now().isoformat(timespec="seconds"), "provider": args.provider,
        "model": C.MODELS[args.provider]["model"], "prompt": args.prompt, "tag": args.tag,
        "n": len(results), "n_api_calls": len(new), "n_failed": n_fail,
        "wall_sec": round(wall, 1), "tokens_in": tin, "tokens_out": tout, "cost_cny": round(cost, 4),
        "sec_per_1k": round(wall / max(len(new), 1) * 1000, 1) if new else None,
        "cost_per_1k_cny": round(cost / max(len(new), 1) * 1000, 3) if new else None,
    }
    stats_path = C.OUT_DIR / "run_stats.csv"
    pd.DataFrame([stats]).to_csv(stats_path, mode="a", header=not stats_path.exists(), index=False)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"已保存 {out}；失败 {n_fail} 条，见 outputs/failures_{args.provider}_{args.prompt}.jsonl，重跑同一命令即可补齐")


if __name__ == "__main__":
    main()
