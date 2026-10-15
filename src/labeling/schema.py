"""标签体系 v0.2 + 输出校验。

设计原则：
- 模型输出英文代码（稳定、好解析、不受中英混写影响），展示时映射成中文。
- 「方面」是评论在谈什么，「痛点」是具体出了什么问题。一个负面方面通常对应一个痛点，
  例如 aspect=skin_feel/neg + pain_point=breakout。
- 方面情绪只有 pos / neu / neg；评论没提到的方面不输出。
- 购买动机单选，没提到就是 none（大多数评论会是 none，这是正常的）。

v0.1 按美妆常识拟定；v0.2 根据试读 50 条护肤评论修订（见 annotation/pilot_50_notes.md）：
+ 方面 moisturizing；+ 痛点 ingredient_concern / white_cast / pilling；+ 动机 ingredient_driven；
+ 字段 skin_type（多选）、repurchase_intent、is_relevant。
改标签体系时把 SCHEMA_VERSION 加一，旧缓存自动失效。
"""

SCHEMA_VERSION = "0.2"

ASPECTS = {
    "moisturizing":     ("保湿/滋润", "补水、保湿、滋润，或反过来拔干、不够润"),
    "efficacy":         ("功效", "保湿以外的针对性功效：抗老紧致、祛痘、提亮淡斑、控油、清洁力、舒缓湿疹、防晒效果等"),
    "skin_feel":        ("肤感/质地", "上脸/上手的感觉：清爽、油腻、黏、厚重、吸收快慢、顺滑"),
    "scent":            ("气味", "香味好闻、难闻、太浓、无味"),
    "longevity":        ("持久度", "妆效、香味、发型、指甲油等能保持多久"),
    "irritation":       ("刺激/过敏", "对皮肤、头皮、眼睛是否温和，有无刺痛、红痒、过敏、爆痘"),
    "ingredients":      ("成分", "提到成分本身：天然、无香精、含酒精、某成分好/不好"),
    "color_appearance": ("颜色/外观", "色号、显色、与图片是否一致、产品外观"),
    "packaging":        ("包装", "瓶子、泵头、喷头、瓶盖、密封、漏液、破损、外包装、便携性（泵头按不出来归这里）"),
    "ease_of_use":      ("使用便利", "好不好用、操作难易、说明书、工具是否顺手"),
    "price_value":      ("价格/性价比", "贵、便宜、值不值、与同类比较；「一点就够用」算正面"),
    "authenticity":     ("真伪/品质一致", "疑似假货、像被用过、与以往购买的批次不同、配方变了、过期"),
    "shipping":         ("物流/到货", "到货速度、到货状态、错发漏发"),
    "customer_service": ("客服/售后", "退换货、商家沟通、售后处理"),
}

SENTIMENTS = {"pos": ("正", 1), "neu": ("中", 0), "neg": ("负", -1)}

PAIN_POINTS = {
    "allergy_irritation": "过敏/刺激",
    "breakout":           "致痘/闷痘",
    "no_effect":          "没效果",
    "greasy_sticky":      "油腻/黏",
    "drying":             "拔干/干燥",
    "bad_scent":          "气味难闻/太浓",
    "not_lasting":        "不持久",
    "color_mismatch":     "色差/与描述不符",
    "leak_damage":        "漏液/破损",
    "hard_to_use":        "难用",
    "overpriced":         "价格贵/不值",
    "suspected_fake":     "疑似假货/二手/批次异常",
    "expired_spoiled":    "过期/变质",
    "small_quantity":     "量少",
    "ingredient_concern": "成分担忧/成分不透明",
    "white_cast":         "泛白",
    "pilling":            "搓泥",
    "wrong_missing_item": "错发/漏发",
    "slow_shipping":      "物流慢",
    "other":              "其他",
}

PURCHASE_MOTIVES = {
    "repurchase":      "回购（明确说再次购买/一直在用）",
    "gift":            "送礼",
    "recommendation":  "他人推荐/种草（朋友、网红、医生、评论区）",
    "promotion":       "促销/优惠",
    "problem_solving": "为解决特定问题（痘痘、脱发、干皮等）",
    "replacement":     "替代原来用的产品（停产、涨价、想换）",
    "trial":           "尝鲜/试用新品",
    "ingredient_driven": "为某个成分、天然/无添加或某个功效宣称而买",
    "none":            "未提及",
}

OVERALL = ["pos", "neu", "neg"]

SKIN_TYPES = {
    "dry": "干皮", "oily": "油皮", "combination": "混合皮", "sensitive": "敏感肌",
    "acne_prone": "痘痘肌", "mature": "熟龄（提到年龄大、皱纹松弛）",
}

REPURCHASE_INTENT = {"yes": "会再买/已回购", "no": "不会再买", "none": "未提及"}


def validate(obj):
    """校验并规范化模型输出。返回 (clean_obj, errors)。errors 非空的条目进失败日志并重跑。"""
    errors = []
    if not isinstance(obj, dict):
        return None, ["not_a_dict"]

    aspects = []
    seen = set()
    for a in obj.get("aspects") or []:
        if not isinstance(a, dict):
            errors.append("aspect_not_dict"); continue
        name, sent = a.get("aspect"), a.get("sentiment")
        if name not in ASPECTS:
            errors.append(f"bad_aspect:{name}"); continue
        if sent not in SENTIMENTS:
            errors.append(f"bad_sentiment:{sent}"); continue
        if name in seen:  # 同一方面重复出现：保留第一个
            continue
        seen.add(name)
        aspects.append({"aspect": name, "sentiment": sent})

    pains = []
    for p in obj.get("pain_points") or []:
        if p in PAIN_POINTS:
            if p not in pains:
                pains.append(p)
        else:
            errors.append(f"bad_pain_point:{p}")

    motive = obj.get("purchase_motive", "none")
    if motive not in PURCHASE_MOTIVES:
        errors.append(f"bad_motive:{motive}")
        motive = "none"

    overall = obj.get("overall_sentiment")
    if overall not in OVERALL:
        errors.append(f"bad_overall:{overall}")

    skins = []
    for t in obj.get("skin_type") or []:
        if t in SKIN_TYPES:
            if t not in skins:
                skins.append(t)
        else:
            errors.append(f"bad_skin_type:{t}")

    intent = obj.get("repurchase_intent", "none")
    if intent not in REPURCHASE_INTENT:
        errors.append(f"bad_repurchase_intent:{intent}")
        intent = "none"

    relevant = obj.get("is_relevant", True)
    if not isinstance(relevant, bool):
        errors.append(f"bad_is_relevant:{relevant}")
        relevant = True

    clean = {"aspects": aspects, "pain_points": pains, "purchase_motive": motive,
             "overall_sentiment": overall, "skin_type": skins,
             "repurchase_intent": intent, "is_relevant": relevant}
    return clean, errors
