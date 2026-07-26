#!/usr/bin/env python3
"""
简化的 RAG 快速入门示例

展示如何使用 SQCVecDB 构建最小化的 RAG 系统。
这是一个完整的端到端示例，包含：
  1. 文档加载
  2. 向量化和存储
  3. 查询和检索
  4. LLM 提示词生成

运行方式：
    python examples/rag_quickstart.py
"""

import hashlib
from typing import List, Optional

import numpy as np

from sqcvecdb import Collection


# ============================================================================
# 简单的向量化工具
# ============================================================================

class TextEmbedder:
    """简单的文本向量化工具（实际应用中应使用真实的嵌入模型）"""
    
    def __init__(self, dim: int = 256):
        self.dim = dim
    
    def encode(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        # 基于哈希的简单向量化（确保同一文本总是得到相同向量）
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**31))
        vector = np.random.randn(self.dim).astype(np.float32)
        
        # 归一化为单位向量
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


# ============================================================================
# 简单的 RAG 类
# ============================================================================

class SimpleRAG:
    """最小化的 RAG 系统"""
    
    def __init__(self, dim: int = 256, index_type: str = "flat"):
        """
        初始化 RAG 系统
        
        Args:
            dim: 向量维度
            index_type: 索引类型 ('flat', 'ivf', 'hnsw')
        """
        self.embedder = TextEmbedder(dim=dim)
        self.collection = Collection(
            dim=dim,
            metric="cosine",  # 使用余弦距离
            index_type=index_type
        )
        self.doc_id_counter = 0
    
    def add_document(self, title: str, content: str, metadata: Optional[dict] = None) -> None:
        """添加文档到 RAG 系统"""
        self.doc_id_counter += 1
        vec_id = self.doc_id_counter
        
        # 向量化
        embedding = self.embedder.encode(content)
        
        # 存储
        doc_metadata = {
            "title": title,
            "content": content[:500],  # 前500个字符作为摘要
            **(metadata or {})
        }
        
        self.collection.insert(vec_id=vec_id, vector=embedding, metadata=doc_metadata)
        print(f"✓ 已添加文档: {title}")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        """检索相关文档"""
        # 向量化查询
        query_embedding = self.embedder.encode(query)
        
        # 搜索
        results = self.collection.search(query_embedding, top_k=top_k)
        
        # 格式化结果
        retrieved_docs = []
        for vec_id, distance in results:
            vector_data = self.collection.get(vec_id)
            if vector_data is not None:
                _, metadata = vector_data
                retrieved_docs.append({
                    "id": vec_id,
                    "title": metadata.get("title", "Unknown"),
                    "content": metadata.get("content", ""),
                    "similarity": 1 - distance  # 转换为相似度
                })
        
        return retrieved_docs
    
    def generate_prompt(self, query: str, top_k: int = 3) -> str:
        """生成用于 LLM 的提示词"""
        # 检索相关文档
        docs = self.retrieve(query, top_k=top_k)
        
        # 构建上下文
        context = "\n\n".join([
            f"【文档 {i+1}】{doc['title']}\n{doc['content']}"
            for i, doc in enumerate(docs)
        ])
        
        # 生成提示词
        prompt = f"""请根据以下背景信息回答问题。如果背景信息不足，请说"信息不足"。

【背景信息】
{context}

【问题】
{query}

【回答】
"""
        return prompt


# ============================================================================
# 演示函数
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎯 SQCVecDB RAG 快速入门示例".center(70))
    print("="*70 + "\n")
    
    # ========== 第一步：创建 RAG 系统 ==========
    print("📦 第一步：初始化 RAG 系统")
    print("-" * 70)
    rag = SimpleRAG(dim=256, index_type="ivf")
    print("✓ RAG 系统已创建\n")
    
    # ========== 第二步：添加文档 ==========
    print("📚 第二步：添加示例文档")
    print("-" * 70)
    
    documents = [
        {
            "title": "什么是向量数据库",
            "content": "向量数据库是一种为高维向量优化的数据库。它使用高效的索引算法（如 IVF、HNSW）来加速相似性搜索。"
        },
        {
            "title": "RAG 技术原理",
            "content": "RAG 结合检索和生成：首先从知识库检索相关文档，然后将其作为上下文传给 LLM，帮助生成更准确的答案。"
        },
        {
            "title": "Python 最佳实践",
            "content": "Python 代码应该遵循 PEP 8 风格指南，使用类型注解提高代码可维护性，并编写充分的单元测试。"
        },
        {
            "title": "机器学习基础",
            "content": "机器学习包括监督学习和无监督学习。向量表示（embedding）是现代 ML 系统的核心组件。"
        },
    ]
    
    for doc in documents:
        rag.add_document(doc["title"], doc["content"])
    print()
    
    # ========== 第三步：执行查询 ==========
    print("🔍 第三步：执行查询并检索相关文档")
    print("-" * 70)
    
    queries = [
        "向量数据库有什么作用？",
        "RAG 如何工作？",
        "Python 代码怎样写得更好？"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔹 查询 {i}: {query}")
        
        # 检索文档
        docs = rag.retrieve(query, top_k=2)
        print(f"  检索到 {len(docs)} 个相关文档：")
        
        for j, doc in enumerate(docs, 1):
            print(f"    {j}. [{doc['title']}] (相似度: {doc['similarity']:.2%})")
        
        # 生成提示词
        prompt = rag.generate_prompt(query, top_k=2)
        print(f"\n  📝 生成的 LLM 提示词：")
        print("  " + "-" * 66)
        for line in prompt.strip().split("\n"):
            print(f"  {line}")
        print("  " + "-" * 66 + "\n")
    
    # ========== 第四步：演示不同索引类型 ==========
    print("="*70)
    print("⚡ 演示不同索引类型的性能")
    print("="*70)
    
    test_query = "向量检索的应用场景"
    
    for index_type in ["flat", "ivf", "hnsw"]:
        print(f"\n📊 使用 {index_type.upper()} 索引：")
        
        # 创建新的 RAG 系统
        test_rag = SimpleRAG(dim=256, index_type=index_type)
        for doc in documents:
            test_rag.add_document(doc["title"], doc["content"])
        
        # 执行查询
        results = test_rag.retrieve(test_query, top_k=2)
        for j, doc in enumerate(results, 1):
            print(f"  {j}. {doc['title']} (相似度: {doc['similarity']:.2%})")
    
    # ========== 总结 ==========
    print("\n" + "="*70)
    print("✅ 演示完成！".center(70))
    print("="*70)
    
    print("""
💡 关键要点：

1. 📝 文档管理
   - 添加各种来源的文档
   - 灵活的元数据支持

2. 🔍 检索能力  
   - 快速的相似性搜索
   - 支持多种距离度量

3. 🤖 RAG 集成
   - 生成高质量提示词
   - 易于与 LLM 集成

4. ⚡ 多种索引
   - Flat：精确但较慢
   - IVF：平衡性能
   - HNSW：最快速度

🚀 下一步：
   - 使用真实的嵌入模型（sentence-transformers, OpenAI）
   - 集成真实 LLM (GPT, Claude, Llama)
   - 参考 rag_system_demo.py 了解高级功能
    """)


if __name__ == "__main__":
    main()
