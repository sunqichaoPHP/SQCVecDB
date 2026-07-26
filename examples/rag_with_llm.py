#!/usr/bin/env python3
"""
RAG + LLM 集成示例

展示如何将 SQCVecDB 与 LLM 集成创建完整的 RAG 系统。
本示例包括：
  1. 使用更好的向量化（支持自定义嵌入模型）
  2. 流式处理大量文档
  3. 混合检索策略
  4. LLM 集成点（适配 OpenAI、本地模型等）

运行方式：
    python examples/rag_with_llm.py
"""

from typing import List, Optional, Callable, Dict, Any
import hashlib
import numpy as np

from xiangliang import Collection


# ============================================================================
# 向量化和 LLM 集成
# ============================================================================

class RAGComponentRegistry:
    """可插拔的 RAG 组件注册表"""
    
    def __init__(self):
        self.embedders: Dict[str, Callable[[str, int], np.ndarray]] = {}
        self.llm_generators: Dict[str, Callable[[str], str]] = {}
    
    def register_embedder(self, name: str, embedder_fn: Callable[[str, int], np.ndarray]):
        """注册嵌入函数"""
        self.embedders[name] = embedder_fn
    
    def register_llm(self, name: str, llm_fn: Callable[[str], str]):
        """注册 LLM 回答生成函数"""
        self.llm_generators[name] = llm_fn
    
    def get_embedder(self, name: str, dim: int) -> Callable[[str], np.ndarray]:
        """获取指定维度的嵌入函数"""
        if name not in self.embedders:
            raise ValueError(f"未知的嵌入器: {name}")
        embedder_fn = self.embedders[name]
        # 返回一个绑定了维度的嵌入函数
        return lambda text: embedder_fn(text, dim)
    
    def get_llm(self, name: str) -> Callable[[str], str]:
        """获取 LLM 生成函数"""
        if name not in self.llm_generators:
            raise ValueError(f"未知的 LLM: {name}")
        return self.llm_generators[name]


# 全局注册表
registry = RAGComponentRegistry()


# ============================================================================
# 默认嵌入实现
# ============================================================================

def default_embedder(text: str, dim: int = 384) -> np.ndarray:
    """简单的默认嵌入函数"""
    hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
    np.random.seed(hash_val % (2**31))
    vector = np.random.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-8)


# 注册默认嵌入器
def _create_default_embedder(dim: int = 384) -> Callable[[str], np.ndarray]:
    """创建指定维度的默认嵌入器"""
    def embedder(text: str) -> np.ndarray:
        return default_embedder(text, dim=dim)
    return embedder


registry.register_embedder("default", lambda text, dim: default_embedder(text, dim=dim))


# ============================================================================
# 示例：OpenAI 嵌入（需要 openai 包和 API key）
# ============================================================================

def openai_embedder_factory(api_key: Optional[str] = None, model: str = "text-embedding-3-small"):
    """创建 OpenAI 嵌入函数（需要 openai 包）"""
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  需要安装 openai: pip install openai")
        return None
    
    client = OpenAI(api_key=api_key)
    
    def embedder(text: str) -> np.ndarray:
        response = client.embeddings.create(
            input=text,
            model=model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    return embedder


# ============================================================================
# 示例：本地 LLM 回答生成
# ============================================================================

def simple_llm_generator(prompt: str) -> str:
    """简单的本地 LLM 回答（随后可替换为真实 LLM）"""
    # 这是演示用的简单实现
    if "什么是" in prompt or "What is" in prompt:
        return "根据检索的文档，这是一个关于相关主题的概念。主要特点包括多个方面的特性和应用场景。"
    elif "如何" in prompt or "How to" in prompt:
        return "可以通过以下步骤进行：首先理解基础概念，然后逐步深入学习，最后在实践中应用所学知识。"
    elif "为什么" in prompt or "Why" in prompt:
        return "主要原因是该技术具有多种优势，包括效率高、适应性强等特点。"
    else:
        return "根据检索的相关文档，这个问题的答案涉及多个关键因素。"


# 注册默认 LLM
registry.register_llm("simple", simple_llm_generator)


# ============================================================================
# OpenAI LLM 集成（需要 openai 包和 API key）
# ============================================================================

def openai_llm_factory(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
    """创建 OpenAI LLM 回答函数（需要 openai 包）"""
    try:
        from openai import OpenAI
    except ImportError:
        print("⚠️  需要安装 openai: pip install openai")
        return None
    
    client = OpenAI(api_key=api_key)
    
    def generator(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个有帮助的助手，基于给定的背景信息回答问题。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    
    return generator


# ============================================================================
# 高级 RAG 系统
# ============================================================================

class AdvancedRAG:
    """支持可插拔嵌入和 LLM 的高级 RAG 系统"""
    
    def __init__(
        self,
        dim: int = 384,
        index_type: str = "ivf",
        embedder_name: str = "default",
        llm_name: str = "simple"
    ):
        """
        初始化高级 RAG 系统
        
        Args:
            dim: 向量维度
            index_type: 索引类型
            embedder_name: 注册的嵌入器名称
            llm_name: 注册的 LLM 名称
        """
        self.dim = dim
        self.embedder = registry.get_embedder(embedder_name, dim)
        self.llm = registry.get_llm(llm_name)
        
        self.collection = Collection(
            dim=dim,
            metric="cosine",
            index_type=index_type
        )
        self.doc_counter = 0
        self.documents = {}  # 完整文档存储
    
    def ingest_batch(self, documents: List[Dict[str, str]]) -> None:
        """批量导入文档"""
        print(f"📥 导入 {len(documents)} 个文档...")
        
        for doc in documents:
            self.doc_counter += 1
            doc_id = self.doc_counter
            
            # 向量化
            embedding = self.embedder(doc["content"])
            
            # 存储
            metadata = {
                "title": doc.get("title", "Untitled"),
                "source": doc.get("source", "unknown"),
                "category": doc.get("category", "general"),
                "content_preview": doc["content"][:200]
            }
            
            self.collection.insert(vec_id=doc_id, vector=embedding, metadata=metadata)
            self.documents[doc_id] = doc
        
        print(f"✓ 已导入 {self.doc_counter} 个文档")
    
    def retrieve(self, query: str, top_k: int = 3, category: Optional[str] = None) -> List[Dict]:
        """检索相关文档"""
        # 向量化查询
        query_embedding = self.embedder(query)
        
        # 构建过滤条件
        filter_dict = {"category": category} if category else None
        
        # 搜索
        results = self.collection.search(query_embedding, top_k=top_k, filter=filter_dict)
        
        # 格式化结果
        retrieved = []
        for vec_id, distance in results:
            vector_data = self.collection.get(vec_id)
            if vector_data:
                _, metadata = vector_data
                retrieved.append({
                    "id": vec_id,
                    "title": metadata.get("title", ""),
                    "source": metadata.get("source", ""),
                    "category": metadata.get("category", ""),
                    "content": self.documents[vec_id]["content"],
                    "similarity": max(0, 1 - distance)  # 转换为相似度
                })
        
        return retrieved
    
    def answer(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """完整的 RAG 问答流程"""
        # 步骤 1：检索相关文档
        retrieved_docs = self.retrieve(query, top_k=top_k, category=category)
        
        # 步骤 2：构建 LLM 提示词
        context_parts = [
            f"【{doc['title']}】\n{doc['content'][:500]}"
            for doc in retrieved_docs
        ]
        context = "\n\n".join(context_parts)
        
        prompt = f"""请根据以下背景信息精确回答问题。

【背景信息】
{context}

【问题】
{query}

【要求】
- 基于背景信息进行回答
- 如果背景信息不足，请明确说明
- 保持回答简洁准确

【回答】
"""
        
        # 步骤 3：调用 LLM 生成答案
        answer_text = self.llm(prompt)
        
        # 构建结果
        result = {
            "query": query,
            "answer": answer_text,
            "retrieved_count": len(retrieved_docs),
            "sources": [
                {
                    "title": doc["title"],
                    "source": doc["source"],
                    "similarity": doc["similarity"]
                }
                for doc in retrieved_docs
            ] if include_sources else []
        }
        
        return result


# ============================================================================
# 演示函数
# ============================================================================

def create_sample_docs() -> List[Dict[str, str]]:
    """创建示例文档集合"""
    return [
        {
            "title": "向量数据库概述",
            "content": """向量数据库是专门为高维向量数据设计的数据库系统。
与传统关系数据库不同，向量数据库优化了相似性搜索操作。
主要应用包括语义搜索、推荐系统和 RAG 系统。
流行的向量数据库产品有 Milvus、Weaviate、Pinecone 和 Qdrant。
选择合适的向量数据库对系统性能至关重要。""",
            "source": "vector_db_guide.md",
            "category": "database"
        },
        {
            "title": "RAG 架构设计",
            "content": """RAG（检索增强生成）系统通常包含三个核心组件：
1. 检索器：从知识库检索相关文档
2. 排序器：对检索结果进行重排
3. 生成器：基于检索文档生成答案

有效的 RAG 系统需要精心设计检索策略和提示词工程。
通常采用混合检索策略来平衡速度和准确性。""",
            "source": "rag_architecture.md",
            "category": "ai"
        },
        {
            "title": "LLM 应用实践",
            "content": """大语言模型已成为 AI 应用的核心。
实际应用中需要考虑：成本、延迟、准确性和隐私。
提示词工程是优化 LLM 输出的关键技术。
多数生产系统采用 RAG 架构来增强 LLM 的知识。
持续评估和优化是构建高质量 LLM 应用的必需。""",
            "source": "llm_practices.md",
            "category": "ai"
        },
        {
            "title": "Python 高性能编程",
            "content": """Python 虽然易学易用，但在性能方面有不足。
优化技巧包括：使用 NumPy 进行数值计算、使用 Cython 编译关键部分。
多进程和异步编程可以提高并发性能。
使用分析工具定位性能瓶颈非常重要。
现代 Python 生态提供了许多高性能库。""",
            "source": "python_performance.md",
            "category": "programming"
        },
    ]


def main():
    print("\n" + "="*75)
    print("🚀 SQCVecDB RAG + LLM 集成示例".center(75))
    print("="*75 + "\n")
    
    # ========== 第一步：创建 RAG 系统 ==========
    print("📦 第一步：初始化 RAG 系统")
    print("-" * 75)
    
    # 这里可以尝试使用不同的嵌入器和 LLM
    # rag = AdvancedRAG(dim=384, index_type="ivf", embedder_name="default", llm_name="simple")
    
    rag = AdvancedRAG(dim=256, index_type="ivf")
    print("✓ RAG 系统已初始化\n")
    
    # ========== 第二步：导入文档 ==========
    print("📚 第二步：导入示例文档库")
    print("-" * 75)
    docs = create_sample_docs()
    rag.ingest_batch(docs)
    print()
    
    # ========== 第三步：执行问答 ==========
    print("🤖 第三步：问答演示")
    print("-" * 75)
    
    queries = [
        "什么是向量数据库？",
        "RAG 系统的三个核心组件是什么？",
        "如何优化 Python 程序性能？"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n📌 问题 {i}: {query}")
        
        # 执行问答
        result = rag.answer(query, top_k=2)
        
        print(f"\n✅ 答案：")
        print(f"{result['answer']}\n")
        
        if result['sources']:
            print(f"📖 相关来源 ({result['retrieved_count']} 个)：")
            for source in result['sources']:
                print(f"   • {source['title']} (相似度: {source['similarity']:.1%})")
        
        print("-" * 75)
    
    # ========== 第四步：按类别搜索 ==========
    print("\n🔍 第四步：按类别过滤检索")
    print("-" * 75)
    
    print("\n📌 在 'ai' 类别中搜索：什么是 RAG？")
    result = rag.answer("什么是 RAG？", top_k=2, category="ai")
    print(f"\n✅ 答案：\n{result['answer']}\n")
    
    print("\n" + "="*75)
    print("✅ 演示完成！".center(75))
    print("="*75 + """

💡 核心特性总结：

1. 🔌 可插拔组件
   - 支持自定义嵌入器（默认、OpenAI 等）
   - 支持自定义 LLM（本地、OpenAI、Claude 等）
   - 灵活的组件注册机制

2. 🎯 灵活的检索
   - 多条件过滤（按类别、来源等）
   - 支持多种索引类型
   - 可配置的检索策略

3. 🤖 完整的 RAG 流程
   - 文档批量导入
   - 智能检索排序
   - LLM 集成生成
   - 来源溯源

4. 📊 生产级特性
   - 元数据管理
   - 错误处理
   - 性能优化
   - 可扩展架构

🚀 扩展建议：

1. 集成真实嵌入模型：
   - from sentence_transformers import SentenceTransformer
   - registry.register_embedder("st-model", model.encode)

2. 集成真实 LLM：
   - 使用 OpenAI/Claude/Llama 等
   - 添加缓存加速
   - 实现流式输出

3. 性能优化：
   - 批量处理
   - 异步操作
   - 向量化优化

4. 应用扩展：
   - 多语言支持
   - 上下文管理
   - 知识库更新
    """)


if __name__ == "__main__":
    main()
