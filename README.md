# 基于大模型的用户评论洞察与决策分析（美妆）

用大模型把 Amazon 护肤评论拆成「方面 × 情绪、痛点、购买动机」结构化标签，用 300 条人工标注评估 AI 准确率，再用回归量化各方面对评分的影响，最后做成 Streamlit 看板和一页决策建议。

数据：Amazon Reviews 2023（McAuley Lab）`Beauty_and_Personal_Care` 类目中的护肤（Skin Care）子品类，2019–2023 年，清洗后随机抽 3 万条做分析主样本。

（最初试过 `All_Beauty`，发现以假睫毛、美甲、假发等长尾小卖家为主，品牌分散、价格缺失 72%、没有子品类，不适合做品牌和品类洞察，所以换成了主类目里的护肤。）

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env            # 填入 DEEPSEEK_API_KEY / DASHSCOPE_API_KEY

python -m src.data_prep --download       # 1. 下载原始数据（几 GB，也可以用浏览器下载到 data/raw/）
python -m src.prefilter                  # 1. 流式筛出护肤子品类 -> data/raw/Skincare.jsonl.gz
python -m src.data_prep                  # 1. 清洗 + 抽样
python -m src.basic_stats                # 1. 基础统计 -> outputs/basic_stats.md
python -m src.eval.make_annotation_sheet # 2. 抽 50 条试读 + 300 条评估集 -> annotation/

# 2. 先用 mock 跑通（不花钱），再上真模型
python -m src.labeling.run_labeling --provider mock --prompt v1 --limit 200

# 3. 评估集：2 模型 × 2 提示词
for p in deepseek qwen; do for v in v1 v2; do
  python -m src.labeling.run_labeling --provider $p --prompt $v --ids data/processed/eval_ids.txt --tag eval
done; done
python -m src.eval.evaluate --pred data/processed/labels_*_eval.parquet

# 全量（选评估最好的方案）
python -m src.labeling.run_labeling --provider qwen --prompt v2
```

## 目录

| 路径 | 作用 |
|---|---|
| `src/config.py` | 路径、清洗参数、模型与单价（跑前核对官网价格） |
| `src/prefilter.py` | 从几 GB 的原始文件里流式筛出护肤商品和 2019 年后的评论（只用标准库） |
| `src/data_prep.py` | 下载、清洗（去空、限时间、去重、去短评、关联品牌价格）、抽样 |
| `src/basic_stats.py` | 清洗漏斗、评分/品牌/价格段分布、季度趋势图 |
| `src/labeling/schema.py` | 标签体系 v0.2（14 个方面、20 个痛点、9 个动机，加肤质、复购意向、是否相关）和输出校验；修订依据见 annotation/pilot_50_notes.md |
| `src/labeling/prompts.py` | v1 规则版 / v2 规则 + 难例示例版 |
| `src/labeling/llm_client.py` | OpenAI 兼容调用、SQLite 缓存、指数退避重试、格式错误重问、失败日志 |
| `src/labeling/run_labeling.py` | 并发批量标注、宽表输出、每千条耗时与费用记录、百炼 Batch 文件生成与导入 |
| `src/eval/make_annotation_sheet.py` | 试读 50 条 + 按星级分层抽 300 条，生成带下拉框的 Excel |
| `src/eval/evaluate.py` | 方面 P/R/F1、方面情绪一致率、联合 F1、痛点/动机/整体情绪、难例分组 |

## 设计说明

- 模型只看评论标题和正文，不看星级、商品名、品牌，避免「看答案」；星级留给后面的回归当因变量。
- 标签用英文代码输出（稳定、好解析），展示时映射成中文。
- 缓存 key 包含模型、提示词版本、schema 版本和评论内容，改了提示词或标签体系会自动重新标注，其他情况重跑不花钱。
- 评估集按星级等量抽样（每档 60 条），否则差评太少、痛点没法评估；整体情绪准确率另给按真实星级分布加权的版本。
