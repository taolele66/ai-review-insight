"""调用大模型：缓存 + 重试 + JSON 校验 + 失败日志 + token/费用记录。

- 缓存：SQLite，key = hash(模型, 提示词版本, schema 版本, 评论内容)。重跑时命中缓存不再花钱。
- 重试：网络错误/限流用指数退避重试；输出不是合法 JSON 或校验不通过时，再请求一次（temperature=0 下
  通常第二次就好），仍失败则写入失败日志，不进缓存，下次重跑会再试。
"""
import hashlib
import json
import os
import random
import re
import sqlite3
import threading
import time

from src import config as C
from src.labeling.prompts import build_messages
from src.labeling.schema import SCHEMA_VERSION, validate

try:
    from dotenv import load_dotenv
    load_dotenv(C.ROOT / ".env")
except ImportError:
    pass


class Cache:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def get(self, key):
        with self.lock:
            row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key, value):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, json.dumps(value, ensure_ascii=False)))
            self.conn.commit()


def parse_json(text):
    """模型偶尔会包 ```json 代码块或在前后加话，取第一个 {...}。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise


class Labeler:
    def __init__(self, provider, prompt_version, max_retries=5):
        self.provider = provider
        self.cfg = C.MODELS[provider]
        self.prompt_version = prompt_version
        self.max_retries = max_retries
        self.cache = Cache(C.CACHE_DIR / "llm_cache.sqlite")
        C.OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.fail_log = C.OUT_DIR / f"failures_{provider}_{prompt_version}.jsonl"
        self.fail_lock = threading.Lock()
        self.client = None
        if provider != "mock":
            from openai import OpenAI
            key = os.environ.get(self.cfg["key_env"])
            if not key:
                raise RuntimeError(f"缺少环境变量 {self.cfg['key_env']}，请在 .env 里填写")
            self.client = OpenAI(api_key=key, base_url=self.cfg["base_url"], timeout=60)

    def key(self, title, text):
        # 带上 system prompt 本身的哈希：只改规则文字、不改版本号时也会重新标注，不会误用旧缓存
        system = build_messages(self.prompt_version, "", "")[0]["content"]
        raw = json.dumps([self.cfg["model"], self.prompt_version, SCHEMA_VERSION,
                          hashlib.sha256(system.encode()).hexdigest()[:12], title, text], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    def _call(self, messages):
        if self.client is None:
            return _mock_response(messages[-1]["content"])
        resp = self.client.chat.completions.create(
            model=self.cfg["model"], messages=messages, temperature=0,
            response_format={"type": "json_object"},
        )
        u = resp.usage
        return resp.choices[0].message.content, u.prompt_tokens, u.completion_tokens

    def _call_with_retry(self, messages):
        for attempt in range(self.max_retries):
            try:
                return self._call(messages)
            except Exception as e:  # 网络、限流、5xx 重试；key 错、模型名错、参数错等 4xx 重试也没用，直接抛出
                status = getattr(e, "status_code", None)
                if attempt == self.max_retries - 1 or (status and 400 <= status < 500 and status != 429):
                    raise
                wait = min(2 ** attempt, 30) + random.random()
                print(f"  [{self.provider}] {type(e).__name__}: {e}；{wait:.1f}s 后重试")
                time.sleep(wait)

    def preflight(self):
        """正式开跑前先调一次，key、模型名、网络有问题时立刻报错退出，不用等几百条都失败。"""
        if self.client is None:
            return
        try:
            self._call_with_retry(build_messages(self.prompt_version, "", "Love it, very moisturizing."))
        except Exception as e:
            raise SystemExit(f"[{self.provider}] 试调用失败，先别批量跑：{type(e).__name__}: {e}\n"
                             f"常见原因：.env 里的 key 不对，或模型名 {self.cfg['model']} 你的账号没有"
                             f"（运行 python3 scripts/check_models.py 查看可用模型）")

    def label(self, review_id, title, text):
        """返回 dict：labels（规范化后的标签）、tokens、latency、cached、error。"""
        k = self.key(title, text)
        hit = self.cache.get(k)
        if hit is not None:
            return {**hit, "review_id": review_id, "cached": True, "latency": 0.0}

        messages = build_messages(self.prompt_version, title, text)
        t0 = time.time()
        tok_in = tok_out = 0
        last_err, raw = None, None
        for _ in range(2):  # 格式错误时再给一次机会
            try:
                raw, ti, to = self._call_with_retry(messages)
            except Exception as e:
                last_err = f"api_error:{type(e).__name__}:{e}"
                break
            tok_in += ti; tok_out += to
            try:
                labels, errs = validate(parse_json(raw))
            except Exception as e:
                last_err = f"json_error:{e}"
                continue
            if not errs:
                result = {"labels": labels, "tokens_in": tok_in, "tokens_out": tok_out}
                self.cache.set(k, result)
                return {**result, "review_id": review_id, "cached": False, "latency": time.time() - t0}
            last_err = "schema_error:" + ";".join(errs)

        with self.fail_lock, open(self.fail_log, "a") as f:
            f.write(json.dumps({"review_id": review_id, "error": last_err, "raw": raw}, ensure_ascii=False) + "\n")
        return {"review_id": review_id, "labels": None, "tokens_in": tok_in, "tokens_out": tok_out,
                "cached": False, "latency": time.time() - t0, "error": last_err}

    def cost(self, tokens_in, tokens_out):
        return (tokens_in * self.cfg["price_in"] + tokens_out * self.cfg["price_out"]) / 1e6


# ---- mock：不需要 key，用关键词规则模拟输出，只用来测试流程，不代表任何真实效果 ----
_MOCK_RULES = [
    ("scent", r"smell|scent|fragrance", "bad_scent", r"(bad|awful|strong|terrible|weird) (smell|scent)"),
    ("skin_feel", r"greasy|oily|sticky|texture|absorb|lightweight", "greasy_sticky", r"greasy|oily|sticky"),
    ("irritation", r"rash|burn|irritat|allerg|broke out|breakout|itch", "allergy_irritation", r"rash|burn|irritat|allerg|itch"),
    ("packaging", r"bottle|pump|leak|packag|cap|broken", "leak_damage", r"leak|broken|broke"),
    ("price_value", r"price|expensive|cheap|worth|money", "overpriced", r"expensive|overpriced|not worth"),
    ("shipping", r"shipping|arrived|delivery", "slow_shipping", r"late|slow|took forever"),
    ("authenticity", r"fake|counterfeit|different formula", "suspected_fake", r"fake|counterfeit"),
    ("efficacy", r"work|result|difference|effective", "no_effect", r"(didn'?t|does ?n'?t|did not|does not) work|no difference|nothing"),
]
_NEG = re.compile(r"not|n't|bad|awful|terrible|disappoint|waste|return|broke|leak|worst|hate", re.I)


def _mock_response(user_content):
    t = user_content.lower()
    aspects, pains = [], []
    neg_overall = bool(_NEG.search(t))
    for aspect, pat, pain, pain_pat in _MOCK_RULES:
        if re.search(pat, t):
            bad = bool(re.search(pain_pat, t))
            aspects.append({"aspect": aspect, "sentiment": "neg" if bad else ("neg" if neg_overall else "pos")})
            if bad:
                pains.append(pain)
    motive = ("gift" if "gift" in t else "repurchase" if re.search(r"repurchase|again|another bottle|third", t)
              else "recommendation" if "recommend" in t else "none")
    skin = [k for k, pat in [("dry", r"dry skin"), ("oily", r"oily skin"), ("sensitive", r"sensitive")] if re.search(pat, t)]
    intent = ("no" if re.search(r"won'?t buy again|never again", t) else
              "yes" if re.search(r"buy (it )?again|repurchase|ordered (it )?twice", t) else "none")
    out = {"aspects": aspects, "pain_points": pains, "purchase_motive": motive, "skin_type": skin,
           "repurchase_intent": intent, "is_relevant": True,
           "overall_sentiment": "neg" if neg_overall else "pos"}
    n = len(user_content) // 4
    return json.dumps(out), 900 + n, 60
