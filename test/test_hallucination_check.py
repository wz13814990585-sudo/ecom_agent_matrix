import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ecom_agent_matrix.modules.rag.hallucination_check import (
    dynamic_threshold,
    filter_irrelevant_chunks,
    keyword_overlap_score,
    text_relevance_score,
)
from ecom_agent_matrix.modules.rag.lexicon import tokenize


def main():
    short_query = "半袖"
    long_query = "这款可以适合夏天海边穿的平价短袖连衣裙推荐一下"
    relevant = "夏季短袖连衣裙，纯棉透气，适合沙滩度假。"
    irrelevant = "双人露营帐篷，防风防雨，周末出行首选。"

    print("分词(短):", tokenize(short_query))
    print("分词(长,已去停用词):", tokenize(long_query))
    print("半袖 vs 短袖文档 关键词分:", round(keyword_overlap_score(short_query, relevant), 4))
    print("半袖 vs 帐篷文档 关键词分:", round(keyword_overlap_score(short_query, irrelevant), 4))
    print("短query阈值:", dynamic_threshold(short_query), "长query阈值:", dynamic_threshold(long_query))

    chunks = [
        {"chunk_text": relevant, "goods_sku": "SKU-DRESS-002"},
        {"chunk_text": irrelevant, "goods_sku": "SKU-TENT-005"},
    ]
    kept = filter_irrelevant_chunks(short_query, chunks)
    print("综合打分过滤后保留:", [(c["goods_sku"], c["relevance_score"], c.get("keyword_score")) for c in kept])
    print("短query综合分(相关):", round(text_relevance_score(short_query, relevant), 4))


if __name__ == "__main__":
    main()
