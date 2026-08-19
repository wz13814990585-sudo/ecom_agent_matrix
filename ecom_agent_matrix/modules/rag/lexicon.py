"""电商 RAG 词表：分词、停用词、同义词扩展。"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable

# 无意义虚词：会稀释关键词重合度
STOPWORDS = {
    "的", "了", "是", "在", "有", "和", "与", "或", "也", "都", "很", "非常",
    "比较", "可以", "适合", "这款", "这", "款", "这个", "那个", "一种", "一个",
    "一些", "一下", "以及", "等等", "什么", "怎么", "如何", "哪个", "哪些",
    "请问", "帮我", "看看", "推荐", "吗", "呢", "啊", "吧", "呀", "哦", "穿",
    "the", "a", "an", "is", "are", "for", "to", "of", "and", "or", "with",
    "this", "that", "please", "can", "could",
}

# 商品同义词：任一词命中，组内其他词一并计分（解决「半袖 / 短袖」）
SYNONYM_GROUPS: list[set[str]] = [
    {"半袖", "短袖", "短袖上衣", "t恤", "t-shirt", "tee"},
    {"长袖", "长袖上衣"},
    {"连衣裙", "裙子", "skirt", "dress", "长裙"},
    {"海边", "沙滩", "海滩", "beach"},
    {"防水", "防泼水", "waterproof"},
    {"户外", "outdoor", "徒步", "hiking"},
    {"背包", "双肩包", "书包", "backpack", "bag"},
    {"平价", "便宜", "性价比", "affordable", "cheap"},
    {"防晒", "防紫外线", "uv", "upf"},
    {"帐篷", "tent"},
    {"跑鞋", "运动鞋", "trail", "shoes"},
    {"遮阳帽", "太阳帽", "hat", "cap"},
    {"库存", "现货", "缺货", "stock"},
    {"退货", "退款", "returns", "refund"},
]


@lru_cache(maxsize=1)
def _synonym_index() -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        normalized = {w.lower() for w in group}
        for word in normalized:
            index[word] = normalized
    return index


def expand_synonyms(tokens: Iterable[str]) -> set[str]:
    """把每个词扩展成同义词集合，匹配时一并算分。"""
    index = _synonym_index()
    expanded: set[str] = set()
    for token in tokens:
        key = token.lower()
        expanded.add(key)
        expanded.update(index.get(key, set()))
    return expanded


@lru_cache(maxsize=1)
def _init_jieba() -> bool:
    try:
        import jieba  # type: ignore
    except ImportError:
        return False
    for group in SYNONYM_GROUPS:
        for word in group:
            if re.search(r"[\u4e00-\u9fff]", word):
                jieba.add_word(word)
    return True


def _jieba_cut(text: str) -> list[str]:
    if not _init_jieba():
        return []
    import jieba  # type: ignore

    return [w.strip() for w in jieba.lcut(text) if w.strip()]


def tokenize(text: str) -> list[str]:
    """中英混合分词：优先 jieba，否则按中文连续字块 + 英文单词切分。"""
    if not text:
        return []
    text = text.lower().strip()
    words = _jieba_cut(text)
    if not words:
        words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?|[\u4e00-\u9fff]+", text)
        # 无 jieba 时，把连续中文再切成 2 字窗口，提高短词命中
        expanded: list[str] = []
        for w in words:
            if re.fullmatch(r"[\u4e00-\u9fff]{3,}", w):
                expanded.extend(w[i : i + 2] for i in range(len(w) - 1))
                expanded.append(w)
            else:
                expanded.append(w)
        words = expanded
    return [w for w in words if w and w not in STOPWORDS and not re.fullmatch(r"\W+", w)]
