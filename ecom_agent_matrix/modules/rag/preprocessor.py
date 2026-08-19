"""文档清洗、分层分块。"""
# modules/rag/preprocessor.py
import re
from typing import List

# 分块超参（适配bge embedding模型）
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

def clean_text(raw_text: str) -> str:
    """清洗商品原始文本"""
    if not raw_text:
        return ""
    # 去除html标签
    text = re.sub(r"<.*?>", "", raw_text)
    # 去除链接
    text = re.sub(r"http[s]?://\S+", "", text)
    # 去除多余换行、空格
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_chunk(text: str) -> List[str]:
    """滑动窗口分块"""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append(chunk)
        # 滑动窗口，保留重叠上下文
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def process_goods_text(raw_multi_text: str) -> List[str]:
    """对外统一处理入口：清洗 + 分块"""
    clean = clean_text(raw_multi_text)
    return split_chunk(clean)