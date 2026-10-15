#!/usr/bin/env bash
# 第一次在 Mac 上跑之前执行一次：建虚拟环境、装依赖、检查 .env
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo "依赖已装好。请用文本编辑器打开 .env 填入两个 key，然后运行："
echo "  source .venv/bin/activate && bash scripts/run_eval_labeling.sh"
