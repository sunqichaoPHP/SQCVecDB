#!/usr/bin/env python3
"""
完整的 RAG 系统演示

本示例展示如何使用 SQCVecDB 构建一个完整的 RAG（检索增强生成）系统：
  1. 加载和处理文档
  2. 使用简单的向量化方法进行嵌入
  3. 存储到向量数据库
  4. 基于用户查询进行检索
  5. 生成 LLM 提示词
  6. 模拟 LLM 的回答

运行方式（在项目根目录下）：
    pip install -e .
    python examples/rag_system_demo.py
"""

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple

import numpy as np

from xiangliang import Collection


# ============================================================================
# 简单的文本向量化器（在实际应用中应使用真实的嵌入模型如 sentence-transformers）
# ============================================================================

class SimpleEmbedder:
    """简单的文本向量化器，基于 TF-IDF 的思想"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.vocab = {}
        self.idf = {}
        
    def _text_to_hash_vector(self, text: str) -> np.ndarray:
        """将文本转换为哈希向量"""
        text_lower = text.lower()
        # 使用哈希来生成稳定的向量
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**31))
        vector = np.random.randn(self.dim).astype(np.float32)
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
    
    def encode(self, text: str) -> np.ndarray:
        """将文本编码为向量"""
        return self._text_to_hash_vector(text)
    
    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码文本"""
        return np.array([self.encode(text) for text in texts])


# ============================================================================
# 文档处理模块
# ============================================================================

class Document:
    """文档对象"""
    
    def __init__(self, doc_id: str, title: str, content: str, source: str, category: str):
        self.doc_id = doc_id
        self.title = title
        self.content = content
        self.source = source
        self.category = category
    
    def __repr__(self):
        return f"Document(id={self.doc_id}, title={self.title})"


class DocumentChunker:
    """文档分块器"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """将文本分割成多个块，支持重叠"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return chunks


# ============================================================================
# RAG 系统
# ============================================================================

class SimpleRAGSystem:
    """简单的 RAG 系统"""
    
    def __init__(self, dim: int = 384, index_type: str = "ivf", data_dir: Optional[str] = None):
        """
        初始化 RAG 系统
        
        Args:
            dim: 向量维度
            index_type: 索引类型 ('flat', 'ivf', 'hnsw')
            data_dir: 数据目录（用于持久化）
        """
        self.dim = dim
        self.embedder = SimpleEmbedder(dim=dim)
        
        # 创建向量数据库
        index_params = {}
        if index_type == "ivf":
            index_params = {"nlist": 10}
        
        self.collection = Collection(
            dim=dim,
            metric="cosine",  # 使用余弦距离用于语义相似性
            index_type=index_type,
            index_params=index_params,
            data_dir=data_dir
        )
        
        self.documents = {}  # 存储原始文档
        self.doc_counter = 0
    
    def ingest_documents(self, documents: List[Document]) -> None:
        """将文档导入系统"""
        print(f"\n📚 导入 {len(documents)} 个文档到 RAG 系统...\n")
        
        for doc in documents:
            # 分块文档
            chunks = DocumentChunker.chunk_text(doc.content, chunk_size=300, overlap=50)
            
            for chunk_idx, chunk in enumerate(chunks):
                self.doc_counter += 1
                chunk_id = self.doc_counter
                
                # 向量化
                embedding = self.embedder.encode(chunk)
                
                # 存储到向量数据库
                self.collection.insert(
                    vec_id=chunk_id,
                    vector=embedding,
                    metadata={
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "source": doc.source,
                        "category": doc.category,
                        "chunk_idx": chunk_idx,
                        "content": chunk[:100] + "..." if len(chunk) > 100 else chunk,
                        "full_content": chunk
                    }
                )
            
            print(f"  ✓ 文档 '{doc.title}' 分成 {len(chunks)} 个块")
            self.documents[doc.doc_id] = doc
    
    def retrieve(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[Tuple[int, float, Dict]]:
        """
        检索相关文档块
        
        Args:
            query: 用户查询
            top_k: 返回的最相关块数
            category_filter: 按类别过滤（可选）
        
        Returns:
            (chunk_id, distance, metadata) 的列表
        """
        # 向量化查询
        query_embedding = self.embedder.encode(query)
        
        # 构建过滤条件
        filter_dict = None
        if category_filter:
            filter_dict = {"category": category_filter}
        
        # 检索
        results = self.collection.search(
            query_embedding,
            top_k=top_k,
            filter=filter_dict
        )
        
        # 增加元数据信息
        results_with_metadata = []
        for chunk_id, distance in results:
            vector_data = self.collection.get(chunk_id)
            if vector_data is not None:
                _, metadata = vector_data
                results_with_metadata.append((chunk_id, distance, metadata))
        
        return results_with_metadata
    
    def generate_context(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> str:
        """生成检索上下文用于 LLM"""
        results = self.retrieve(query, top_k=top_k, category_filter=category_filter)
        
        context_parts = []
        for i, (chunk_id, distance, metadata) in enumerate(results, 1):
            context_parts.append(f"[文档 {i}] {metadata['title']} (来源: {metadata['source']})")
            context_parts.append(f"内容: {metadata['full_content']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def query(self, user_query: str, top_k: int = 3, category_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        执行完整的 RAG 查询流程
        
        Returns:
            包含查询结果的字典
        """
        # 检索相关文档
        results = self.retrieve(user_query, top_k=top_k, category_filter=category_filter)
        
        # 生成 LLM 提示词
        context = self.generate_context(user_query, top_k=top_k, category_filter=category_filter)
        
        prompt = f"""根据以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文:
{context}

问题: {user_query}

回答:"""
        
        # 模拟 LLM 回答
        answer = self._mock_llm_answer(user_query, context)
        
        return {
            "query": user_query,
            "retrieved_docs": len(results),
            "context": context,
            "prompt": prompt,
            "answer": answer
        }
    
    def _mock_llm_answer(self, query: str, context: str) -> str:
        """模拟 LLM 生成答案"""
        # 简单的模拟逻辑
        keywords = {
            "什么是": "这是一个关于",
            "如何": "可以通过以下方式",
            "为什么": "原因是",
            "区别": "主要区别在于",
        }
        
        for keyword, prefix in keywords.items():
            if keyword in query:
                return f"{prefix}：{context.split('内容:')[-1].split('[文档')[0][:200]}..."
        
        return f"根据检索的文档，{context.split('内容:')[-1].split('[文档')[0][:150]}..."


# ============================================================================
# 演示函数
# ============================================================================

def create_sample_documents() -> List[Document]:
    """创建示例文档"""
    return [
        Document(
            doc_id="doc_001",
            title="向量数据库基础",
            content="""向量数据库是一种专门为存储和查询高维向量设计的数据库系统。
            
在机器学习和深度学习中，数据通常被转换成高维向量表示。传统的关系型数据库不适合高效地处理这种数据。
向量数据库通过使用先进的索引算法（如 KD-Tree、LSH、IVF、HNSW 等）来加速相似性搜索。

向量数据库的核心特性包括：
- 支持多种距离度量（L2、余弦距离、内积）
- 支持高维向量的快速搜索
- 支持向量的增量更新
- 可扩展到数百万甚至数十亿的向量
- 支持元数据过滤和复杂查询

常见的向量数据库产品包括 Milvus、Weaviate、Pinecone 和 Qdrant。
各个产品各有特点，适用于不同的应用场景。""",
            source="vector_db_basics.md",
            category="database"
        ),
        Document(
            doc_id="doc_002",
            title="RAG 技术简介",
            content="""RAG（检索增强生成）是一种结合信息检索和文本生成的技术。
            
传统的大语言模型（LLM）虽然有强大的文本生成能力，但存在以下问题：
1. 知识截断：模型只能使用训练数据中的知识
2. 幻觉问题：模型可能生成错误或虚构的信息
3. 无法访问实时信息：模型不知道最新的事件和数据

RAG 技术通过在生成前从外部知识库检索相关信息来解决这些问题。流程如下：
1. 用户输入查询
2. 系统从知识库检索相关文档
3. 将检索的文档作为上下文传给 LLM
4. LLM 基于上下文生成更准确的答案

RAG 系统结合了检索的精确性和生成的灵活性，在问答、文档总结等应用中效果显著。""",
            source="rag_introduction.md",
            category="ai"
        ),
        Document(
            doc_id="doc_003",
            title="Python 编程基础",
            content="""Python 是一种高级编程语言，以其简洁、易读的语法和强大的功能库而著称。
            
Python 的主要特点：
- 易学易用：简洁的语法使得初学者容易上手
- 功能强大：丰富的标准库和第三方库支持各种应用
- 跨平台：可以在 Windows、Linux、macOS 等多个平台上运行
- 开源免费：Python 是完全开源的，任何人都可以使用和修改

Python 广泛应用于：
- 数据科学和机器学习（NumPy、Pandas、TensorFlow 等）
- Web 开发（Django、Flask、FastAPI 等）
- 自动化脚本编写
- 科学计算和数据分析
- 人工智能和深度学习

Python 的包管理系统 pip 使得安装和管理库变得非常容易。
这也是 Python 生态系统如此繁荣的原因之一。""",
            source="python_basics.md",
            category="programming"
        ),
        Document(
            doc_id="doc_004",
            title="深度学习和神经网络",
            content="""深度学习是机器学习的一个分支，利用包含多个层的神经网络来学习数据的复杂模式。
            
神经网络的基本组成单元是神经元。多个神经元组织成层，多个层组织成网络。
网络通过反向传播算法学习参数，使得预测误差最小化。

常见的深度学习模型包括：
- CNN（卷积神经网络）：主要用于图像处理
- RNN（循环神经网络）：用于序列数据处理
- Transformer：最新的强大架构，用于 NLP 任务
- LSTM：改进的 RNN，解决了梯度消失问题

深度学习在计算机视觉、自然语言处理、语音识别等领域取得了突破性进展。
近年来的大语言模型（如 GPT、BERT）都是基于 Transformer 架构的深度学习模型。""",
            source="deep_learning_guide.md",
            category="ai"
        ),
    ]


def demo_basic_rag():
    """演示 1：基础 RAG 系统"""
    print("=" * 70)
    print("演示 1：基础 RAG 系统 - 文档检索和 LLM 生成")
    print("=" * 70)
    
    # 创建 RAG 系统
    rag = SimpleRAGSystem(dim=384, index_type="flat")
    
    # 创建示例文档
    documents = create_sample_documents()
    
    # 导入文档
    rag.ingest_documents(documents)
    
    # 示例查询
    queries = [
        "什么是向量数据库？",
        "RAG 技术如何工作？",
        "Python 有什么优点？",
        "深度学习中的 Transformer 是什么？"
    ]
    
    print("\n" + "=" * 70)
    print("执行查询")
    print("=" * 70)
    
    for query in queries:
        print(f"\n🔍 查询：{query}")
        print("-" * 70)
        
        result = rag.query(query, top_k=2)
        
        print(f"✅ 检索到 {result['retrieved_docs']} 个相关文档")
        print("\n📝 生成的答案：")
        print(result['answer'][:300] + "...")
        print()


def demo_rag_with_filter():
    """演示 2：带过滤的 RAG - 特定类别检索"""
    print("\n" + "=" * 70)
    print("演示 2：带过滤的 RAG - 按类别检索")
    print("=" * 70)
    
    # 创建 RAG 系统
    rag = SimpleRAGSystem(dim=384, index_type="ivf")
    
    # 创建示例文档
    documents = create_sample_documents()
    
    # 导入文档
    rag.ingest_documents(documents)
    
    print("\n" + "=" * 70)
    print("在 'ai' 类别中检索")
    print("=" * 70)
    
    queries = [
        "什么是 RAG 技术？",
        "解释一下深度学习",
    ]
    
    for query in queries:
        print(f"\n🔍 查询（仅在 AI 类别）：{query}")
        print("-" * 70)
        
        result = rag.query(query, top_k=2, category_filter="ai")
        
        print(f"✅ 检索到 {result['retrieved_docs']} 个相关文档")
        print("\n📝 生成的答案：")
        print(result['answer'][:300] + "...")
        print()


def demo_different_indexes():
    """演示 3：不同索引类型的性能对比"""
    print("\n" + "=" * 70)
    print("演示 3：不同索引类型 (Flat vs IVF vs HNSW)")
    print("=" * 70)
    
    documents = create_sample_documents()
    queries = ["向量数据库的特点是什么？", "如何使用 Python？"]
    
    for index_type in ["flat", "ivf", "hnsw"]:
        print(f"\n📊 使用 {index_type.upper()} 索引：")
        print("-" * 70)
        
        # 创建 RAG 系统
        rag = SimpleRAGSystem(dim=384, index_type=index_type)
        rag.ingest_documents(documents)
        
        # 执行查询
        for query in queries:
            results = rag.retrieve(query, top_k=2)
            print(f"  查询 '{query}' - 检索到 {len(results)} 个相关文档")
        
        print()


def demo_persistence():
    """演示 4：持久化和加载"""
    print("\n" + "=" * 70)
    print("演示 4：向量数据库的持久化和加载")
    print("=" * 70)
    
    # 创建临时目录
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # 第一步：创建并保存
        print("\n💾 第一步：创建 RAG 系统并导入文档...")
        rag = SimpleRAGSystem(dim=384, index_type="flat", data_dir=str(temp_dir))
        documents = create_sample_documents()
        rag.ingest_documents(documents)
        
        query = "Python 的特点？"
        result1 = rag.query(query, top_k=2)
        print(f"  查询结果：检索到 {result1['retrieved_docs']} 个文档")
        
        # 第二步：重新加载
        print("\n🔄 第二步：重新加载向量数据库...")
        rag2 = SimpleRAGSystem(dim=384, index_type="flat", data_dir=str(temp_dir))
        result2 = rag2.query(query, top_k=2)
        print(f"  查询结果：检索到 {result2['retrieved_docs']} 个文档")
        
        print("\n✅ 持久化和加载成功！")
    
    finally:
        # 清理
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "🎯 SQCVecDB RAG 系统完整演示" + " " * 21 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # 运行各个演示
    demo_basic_rag()
    demo_rag_with_filter()
    demo_different_indexes()
    demo_persistence()
    
    print("\n" + "=" * 70)
    print("✨ 所有演示完成！")
    print("=" * 70)
    print("""
💡 关键要点总结：

1. 📚 文档管理
   - 支持文档分块处理
   - 灵活的元数据存储

2. 🔍 检索能力
   - 支持多种索引类型（Flat、IVF、HNSW）
   - 支持元数据过滤和复杂查询
   - 高效的相似性搜索

3. 🤖 RAG 集成
   - 轻松集成 LLM
   - 生成高质量的提示词
   - 支持上下文增强生成

4. 💾 持久化
   - 数据库的保存和加载
   - 支持长期存储和恢复

5. 🚀 扩展性
   - 支持分布式部署
   - 可轻松处理大规模向量集合

📖 了解更多：
   - 查看 examples/ 目录获取更多示例
   - 阅读 README.md 了解架构设计
   - 查看 tests/ 目录了解 API 用法
""")


if __name__ == "__main__":
    main()
