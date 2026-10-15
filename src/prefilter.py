"""第 1 步（前置）：从 Beauty_and_Personal_Care 大文件里筛出护肤子品类。

原始评论文件有几 GB、两千多万条，pandas 一次读不进内存，这里只用标准库流式读写：
    1. 读 meta，保留 categories[1] == "Skin Care" 的商品（已有 meta_Skincare.jsonl.gz 时跳过）
    2. 读评论，保留属于这些商品、且时间不早于 START_DATE 的评论，只留需要的字段
输出和 All_Beauty 同格式的 data/raw/Skincare.jsonl.gz、meta_Skincare.jsonl.gz，之后 data_prep 照常跑。

为了快：先用字符串切片取 parent_asin，只有命中的行才做 json 解析；有 zcat 时用 zcat 解压（比 gzip 模块快）。

用法：python3 -m src.prefilter
"""
import gzip
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone

from src import config as C

SRC = C.SOURCE_CATEGORY
KEEP_TOP = "Skin Care"
REVIEW_FIELDS = ["rating", "title", "text", "asin", "parent_asin", "user_id", "timestamp",
                 "helpful_vote", "verified_purchase"]
KEY = '"parent_asin": "'


def open_text(path):
    if shutil.which("zcat"):
        p = subprocess.Popen(["zcat", str(path)], stdout=subprocess.PIPE, bufsize=1 << 20)
        return p.stdout, p
    return gzip.open(path, "rb"), None


def filter_meta(log):
    keep = set()
    n = 0
    src, proc = open_text(C.RAW_DIR / f"meta_{SRC}.jsonl.gz")
    with gzip.open(C.RAW_DIR / "meta_Skincare.jsonl.gz", "wt", encoding="utf-8") as fout:
        for line in src:
            n += 1
            m = json.loads(line)
            cats = m.get("categories") or []
            if len(cats) > 1 and cats[1] == KEEP_TOP:
                keep.add(m["parent_asin"])
                m.pop("images", None); m.pop("videos", None)
                fout.write(json.dumps(m, ensure_ascii=False) + "\n")
    if proc:
        proc.wait()
    log["meta_items_all"] = n
    return keep


def main():
    t0 = time.time()
    log = {}
    start_ms = int(datetime.fromisoformat(C.START_DATE).replace(tzinfo=timezone.utc).timestamp() * 1000)

    meta_out = C.RAW_DIR / "meta_Skincare.jsonl.gz"
    if meta_out.exists():
        with gzip.open(meta_out, "rt", encoding="utf-8") as f:
            keep = {json.loads(line)["parent_asin"] for line in f}
    else:
        keep = filter_meta(log)
    log["meta_items_skincare"] = len(keep)
    print(f"护肤商品 {len(keep):,} 个（{time.time() - t0:.0f}s）", flush=True)

    n = n_cat = n_date = 0
    src, proc = open_text(C.RAW_DIR / f"{SRC}.jsonl.gz")
    with gzip.open(C.RAW_DIR / "Skincare.jsonl.gz", "wt", encoding="utf-8", compresslevel=3) as fout:
        for line in src:
            n += 1
            if n % 5_000_000 == 0:
                print(f"  已读 {n:,} 条评论，保留 {n_date:,}（{time.time() - t0:.0f}s）", flush=True)
            line = line.decode("utf-8") if isinstance(line, bytes) else line
            i = line.find(KEY) + len(KEY)
            if line[i:line.find('"', i)] not in keep:
                continue
            n_cat += 1
            r = json.loads(line)
            if r["timestamp"] < start_ms:
                continue
            n_date += 1
            out = {k: r.get(k) for k in REVIEW_FIELDS}
            out["images"] = [1] if r.get("images") else []
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
    if proc:
        proc.wait()
    log["reviews_all"] = n
    log["reviews_skincare"] = n_cat
    log["reviews_skincare_since_start"] = n_date
    log["seconds"] = round(time.time() - t0)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    (C.OUT_DIR / "prefilter_log.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
