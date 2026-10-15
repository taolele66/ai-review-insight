#!/usr/bin/env bash
# 在自己电脑上跑：评估集 300 条 × 2 模型 × 2 提示词，共 1200 次调用，结果有缓存，中断后重跑同一命令即可续上。
# 用法（在项目根目录）：bash scripts/run_eval_labeling.sh
set -e
cd "$(dirname "$0")/.."
for provider in deepseek qwen; do
  for prompt in v1 v2; do
    echo "=== $provider / $prompt ==="
    python3 -m src.labeling.run_labeling --provider "$provider" --prompt "$prompt" \
      --ids data/processed/eval_ids.txt --tag eval --workers 4
  done
done
python3 -m src.eval.compare_runs data/processed/labels_*_eval.parquet > /dev/null
echo "全部完成：结果在 data/processed/labels_*_eval.parquet，对比报告在 outputs/model_agreement.md"
