"""两版提示词：v1 只有规则（zero-shot），v2 = v1 + 针对难例的示例（few-shot）。

注意：只给模型评论标题和正文，不给星级、商品名、品牌，避免模型「看答案」。
"""
import json

from src.labeling.schema import ASPECTS, PAIN_POINTS, PURCHASE_MOTIVES, REPURCHASE_INTENT, SKIN_TYPES


def _table(d, with_desc=True):
    rows = []
    for k, v in d.items():
        if isinstance(v, tuple):
            rows.append(f"- {k}: {v[0]}；{v[1]}" if with_desc else f"- {k}: {v[0]}")
        else:
            rows.append(f"- {k}: {v}")
    return "\n".join(rows)


RULES = f"""你是护肤品电商评论的标注员。阅读一条英文用户评论，按以下规则输出结构化标签。

## 方面 aspects（只列出评论中明确谈到的方面，没谈到的不要列）
{_table(ASPECTS)}

每个方面给一个情绪 sentiment：pos（正面）/ neu（中性或褒贬参半）/ neg（负面）。

## 痛点 pain_points（评论中出现的具体问题，可多选，没有就给空列表）
{_table(PAIN_POINTS)}

## 购买动机 purchase_motive（单选；评论没说为什么买就填 none）
{_table(PURCHASE_MOTIVES)}

## 肤质 skin_type（评论者提到的自己的肤质，可多选，没提到给空列表）
{_table(SKIN_TYPES)}

## 复购意向 repurchase_intent
{_table(REPURCHASE_INTENT)}
只有评论明确表达时才填：yes = 说会再买/已经买了好几次/一直在用；no = 明确说不会再买、已退货、换用别的了。
只是不喜欢、没效果、差评，但没说买不买，填 none。

## 是否相关 is_relevant
看的是**商品本身是不是护肤品**：一次性口罩/防护口罩、汽车配件、头发产品、与护肤无关的工具等为 false。
商品是护肤品时一律为 true，即使评论只在说物流、数量不对、包装或客服。

## 整体情绪 overall_sentiment：pos / neu / neg

## 规则
1. 只依据评论文字判断，不要猜测没写出来的内容。
2. 反讽要按真实意思判断（"Great, another bottle that leaked" 是 packaging/neg）。
3. 一条评论可以有多个方面，各自打情绪；同一方面只出现一次。
4. 保湿、滋润、拔干归 moisturizing，不要归 efficacy。
5. 泵头、喷头、瓶盖等功能问题归 packaging。「一点就够用」归 price_value/pos。
6. 「Great product」「Like it」这类没有具体内容的评论，aspects 给空列表，只给整体情绪。
7. 只用上面列出的英文代码，不要自创。
8. 只输出一个 JSON 对象，不要输出任何解释或 markdown 代码块。

## 输出格式
{{"aspects": [{{"aspect": "<代码>", "sentiment": "pos|neu|neg"}}], "pain_points": ["<代码>"], "purchase_motive": "<代码>", "skin_type": ["<代码>"], "repurchase_intent": "yes|no|none", "is_relevant": true, "overall_sentiment": "pos|neu|neg"}}"""


def _ex(aspects, pains=(), motive="none", skin=(), intent="none", relevant=True, overall="pos"):
    return {"aspects": [{"aspect": a, "sentiment": s} for a, s in aspects], "pain_points": list(pains),
            "purchase_motive": motive, "skin_type": list(skin), "repurchase_intent": intent,
            "is_relevant": relevant, "overall_sentiment": overall}


# few-shot 示例：难例（反讽、多方面混杂、很短、动机/肤质/复购、无关评论）
# 前两条改写自试读 50 条（pilot_ids，不在评估集里），其余为人工构造
EXAMPLES = [
    ("评论标题：Great smell! Not so great moisture\n评论正文：I really love this body wash, smells fresh and wakes me up "
     "in the morning. However it is definitely not moisturizing, I think it actually dried out my skin a bit",
     _ex([("scent", "pos"), ("moisturizing", "neg")], pains=["drying"], overall="neu")),
    ("评论标题：has chemical smell\n评论正文：has a chemical smell & skin feels sticky ALL night long. I'll use this up "
     "but I won't buy again",
     _ex([("scent", "neg"), ("skin_feel", "neg")], pains=["bad_scent", "greasy_sticky"], intent="no", overall="neg")),
    ("评论正文：Oh great, the pump broke on day two and half of it leaked all over my bag. Fantastic.",
     _ex([("packaging", "neg")], pains=["leak_damage"], overall="neg")),
    ("评论正文：My third bottle. My dermatologist recommended it for my dry, sensitive skin and nothing else comes "
     "close for the price. A little goes a long way.",
     _ex([("moisturizing", "pos"), ("irritation", "pos"), ("price_value", "pos")], motive="recommendation",
         skin=["dry", "sensitive"], intent="yes")),
    ("评论正文：Love the smell but it left a white cast and pilled under my makeup. Broke me out too.",
     _ex([("scent", "pos"), ("color_appearance", "neg"), ("skin_feel", "neg"), ("irritation", "neg")],
         pains=["white_cast", "pilling", "breakout"], overall="neg")),
    ("评论正文：Bought it for the niacinamide. Contains parabens though, which the listing didn't mention.",
     _ex([("ingredients", "neg")], pains=["ingredient_concern"], motive="ingredient_driven", overall="neg")),
    ("评论正文：Doesn't fit my car's SAE connector, too short.",
     _ex([], relevant=False, overall="neg")),
    ("评论正文：The ear loops are too short, these face masks hurt after an hour of wearing at work.",
     _ex([], relevant=False, overall="neg")),
    ("评论正文：Didn't do a thing for my dark spots. Disappointed.",
     _ex([("efficacy", "neg")], pains=["no_effect"], overall="neg")),
    ("评论正文：Listing said pack of 6, I only received 1 bar.",
     _ex([("shipping", "neg")], pains=["wrong_missing_item"], overall="neg")),
    ("评论正文：Like it",
     _ex([])),
]


def build_user_content(title, text):
    title = (title or "").strip()
    return f"评论标题：{title}\n评论正文：{text}" if title else f"评论正文：{text}"


def build_messages(version, title, text):
    if version == "v1":
        system = RULES
    elif version == "v2":
        shots = "\n\n".join(
            f"{t}\n输出：{json.dumps(o, ensure_ascii=False)}" for t, o in EXAMPLES)
        system = RULES + "\n\n## 示例\n" + shots
    else:
        raise ValueError(f"unknown prompt version {version}")
    return [{"role": "system", "content": system},
            {"role": "user", "content": build_user_content(title, text)}]
