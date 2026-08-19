import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ecom_agent_matrix.modules.rag.preprocessor import process_goods_text

test_desc = """
Summer lightweight beach dress 轻薄海边连衣裙
100% cotton, waterproof, cheap price!
https://xxx.com/product
<size:S/M/L>
"""
if __name__ == "__main__":
    chunks = process_goods_text(test_desc)
    print("分块结果：")
    for idx, c in enumerate(chunks):
        print(f"块{idx+1}: {c}\n")