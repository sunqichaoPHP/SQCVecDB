"""快速上手示例：插入向量、按元数据过滤检索、保存/加载 collection。

运行方式（在项目根目录下）：
    pip install -e .
    python examples/quickstart.py
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from sqcvecdb import Collection


def main() -> None:
    collection = Collection(dim=4, metric="l2")

    rng = np.random.default_rng(seed=42)
    vectors = rng.random((100, 4)).astype(np.float32)
    categories = ["news", "blog", "paper"]

    for i in range(100):
        collection.insert(
            vec_id=i,
            vector=vectors[i],
            metadata={"category": categories[i % len(categories)]},
        )

    query = vectors[0]
    print("全库检索 top5：")
    for vec_id, dist in collection.search(query, top_k=5):
        print(f"  id={vec_id}, distance={dist:.4f}")

    print("\n只在 category=news 中检索 top5：")
    for vec_id, dist in collection.search(query, top_k=5, filter={"category": "news"}):
        print(f"  id={vec_id}, distance={dist:.4f}")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        collection.save(tmp_dir)
        reloaded = Collection.load(tmp_dir)
        print(f"\n保存并重新加载后，collection 大小: {len(reloaded)}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
