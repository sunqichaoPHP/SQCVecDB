#!/usr/bin/env python3
"""
RAG（检索增强生成）集成演示

本示例展示如何使用 SQCVecDB 为 LLM 应用构建 RAG 系统。
流程：
  1. 加载文本文档
  2. 分块并向量化
  3. 存储到向量数据库
  4. 使用用户查询检索相关上下文
  5. 传给 LLM 生成答案
"""

import numpy as np
from xiangliang.collection import Collection


def demo_basic_rag():
    """基础 RAG 演示：文档检索"""
    print("=" * 60)
    print("演示 1: 基础 RAG - 文档检索")
    print("=" * 60)
    
    # 模拟文档库（实际应用中会从真实文件加载）
    documents = [
        {
            "id": "1",
            "text": "Python 是一门高级编程语言，具有简洁、易读的语法。",
            "source": "programming_101.md",
            "category": "programming"
        },
        {
            "id": "2",
            "text": "向量数据库用于存储和搜索高维向量，特别适合语义搜索。",
            "source": "vector_db_guide.md",
            "category": "database"
        },
        {
            "id": "3",
            "text": "LLM（大语言模型）通过 RAG 技术可以访问外部知识库。",
            "source": "llm_basics.md",
            "category": "ai"
        },
        {
            "id": "4",
            "text": "机器学习中的向量表示可以捕捉数据的语义特征。",
            "source": "ml_concepts.md",
            "category": "ai"
        },
    ]
    
    # 创建向量数据库
    col = Collection(
        name="rag_docs",
        dim=10,  # 使用 10 维向量用于演示（实际应该是 384 或 1536）
        index_type="ivf"
    )
    
    # 存储文档（模拟向量化）
    print("\n📝 存储文档到向量数据库...")
    for doc in documents:
        # 实际应用中使用：embedding = embedding_model.encode(doc["text"])
        # 这里用模拟向量
        embedding = np.random.rand(10).tolist()
        
        col.insert(
            id=doc["id"],
            vector=embedding,
            metadata={
                "text": doc["text"],
                "source": doc["source"],
                "category": doc["category"]
            }
        )
        print(f"  ✓ 已存储 doc_{doc['id']}: {doc['source']}")
    
    # 用户查询
    user_queries = [
        "什么是向量数据库？",
        "Python 的特点是什么？",
        "LLM 如何使用外部知识？"
    ]
    
    print("\n🔍 执行查询并检索相关文档...\n")
    for query in user_queries:
        print(f"📌 用户查询: {query}")
        
        # 查询（实际应该向量化）
        query_embedding = np.random.rand(10).tolist()
        results = col.search(query_embedding, top_k=2)
        
        # 构造提示词
        context = "\n".join([
            f"[{i+1}] {metadata['text']} (来源: {metadata['source']})"
            for i, (_, _, metadata) in enumerate(results)
        ])
        
        print(f"📚 检索到的相关文档:")
        print(context)
        print()


def demo_metadata_filtering():
    """元数据过滤演示"""
    print("=" * 60)
    print("演示 2: 元数据过滤 - 分类查询")
    print("=" * 60)
    
    # 创建数据库
    col = Collection(name="filtered_docs", dim=10, index_type="flat")
    
    # 插入不同类别的文档
    docs_data = [
        ("1", "Python 编程基础", "tutorial", "programming"),
        ("2", "深度学习入门", "tutorial", "ai"),
        ("3", "SQL 数据库设计", "guide", "database"),
        ("4", "PyTorch 实战", "guide", "ai"),
        ("5", "Web 开发最佳实践", "best-practice", "programming"),
    ]
    
    print("\n📝 插入不同类别的文档...\n")
    for doc_id, title, doc_type, category in docs_data:
        col.insert(
            id=doc_id,
            vector=np.random.rand(10),
            metadata={
                "title": title,
                "type": doc_type,
                "category": category
            }
        )
        print(f"  ✓ {title} ({category})")
    
    # 按类别查询
    print("\n🏷️ 按类别过滤查询:\n")
    
    query_embedding = np.random.rand(10)
    
    # 查询所有 AI 相关的文档
    print("查询 1: 所有 AI 类别的文档")
    results = col.search(query_embedding, top_k=10)
    ai_docs = [m["title"] for _, _, m in results if m["category"] == "ai"]
    print(f"  找到: {ai_docs}\n")
    
    # 查询所有教程类型的文档
    print("查询 2: 所有教程类型的文档")
    results = col.search(query_embedding, top_k=10)
    tutorials = [m["title"] for _, _, m in results if m["type"] == "tutorial"]
    print(f"  找到: {tutorials}\n")


def demo_rag_pipeline():
    """完整 RAG 管道演示"""
    print("=" * 60)
    print("演示 3: 完整 RAG 管道")
    print("=" * 60)
    
    # 步骤 1: 创建知识库
    print("\n[步骤 1] 创建知识库...")
    col = Collection(name="knowledge_base", dim=10, index_type="ivf")
    
    knowledge_base = [
        {
            "question": "SQCVecDB 是什么？",
            "answer": "SQCVecDB 是一个轻量级向量数据库，专为 RAG 应用设计。",
            "tags": ["vector-db", "rag"]
        },
        {
            "question": "如何使用 SQCVecDB？",
            "answer": "通过 Python SDK 或 REST API 即可轻松集成。",
            "tags": ["usage", "api"]
        },
        {
            "question": "SQCVecDB 支持哪些索引？",
            "answer": "支持 Flat、IVF 和 HNSW 三种索引类型。",
            "tags": ["index", "feature"]
        },
    ]
    
    for i, item in enumerate(knowledge_base):
        col.insert(
            id=str(i),
            vector=np.random.rand(10),
            metadata=item
        )
    print(f"  ✓ 已加载 {len(knowledge_base)} 个知识项")
    
    # 步骤 2: 用户提问
    print("\n[步骤 2] 用户提问...")
    user_question = "如何开始使用这个向量数据库？"
    print(f"  Q: {user_question}")
    
    # 步骤 3: 检索相关知识
    print("\n[步骤 3] 检索相关知识...")
    query_vec = np.random.rand(10)
    results = col.search(query_vec, top_k=2)
    print(f"  找到 {len(results)} 个相关知识项：")
    for i, (doc_id, score, metadata) in enumerate(results):
        print(f"    {i+1}. {metadata['question']}")
        print(f"       答案: {metadata['answer']}")
    
    # 步骤 4: 构建提示词
    print("\n[步骤 4] 构建 LLM 提示词...")
    context = "\n".join([
        f"- {m['question']}: {m['answer']}"
        for _, _, m in results
    ])
    
    llm_prompt = f"""根据以下知识库信息回答用户问题：

知识库:
{context}

用户问题: {user_question}

请给出基于知识库的答案。"""
    
    print("  LLM 将收到以下提示词：")
    print("  " + "-" * 50)
    print("  " + "\n  ".join(llm_prompt.split("\n")))
    print("  " + "-" * 50)
    
    # 步骤 5: 模拟 LLM 响应
    print("\n[步骤 5] LLM 生成答案（模拟）...")
    llm_response = (
        "基于知识库，你可以通过 Python SDK 或 REST API 使用 SQCVecDB。"
        "它支持 Flat、IVF 和 HNSW 三种索引类型，选择合适的索引"
        "可以根据你的数据量和性能需求进行优化。"
    )
    print(f"  A: {llm_response}")


def demo_production_setup():
    """生产环境设置建议"""
    print("=" * 60)
    print("生产环境 RAG 设置指南")
    print("=" * 60)
    
    setup_guide = """
1. 【选择嵌入模型】
   - 轻量级: sentence-transformers/all-MiniLM-L6-v2 (384 维)
   - 高精度: sentence-transformers/all-mpnet-base-v2 (768 维)
   - 多语言: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

2. 【文档预处理】
   - 分块: 300-500 字符/块，重叠 50-100 字符
   - 清理: 去除 HTML 标签、特殊字符
   - 元数据: 保存源文件、页码、标题等

3. 【索引选择】
   - <100K 文档: 使用 Flat 或 IVF (recall ≥ 95%)
   - >100K 文档: 使用 HNSW (QPS 优先)
   - 可调性强: IVF (nlist/nprobe 可调)

4. 【性能优化】
   - 批量插入: insert_many() 而非逐条 insert()
   - 分片部署: 使用分布式客户端跨多个节点
   - 缓存策略: 热门查询缓存

5. 【监控和维护】
   - 定期 checkpoint 创建快照
   - 监控索引统计信息 (get_stats())
   - 定期清理过期文档

6. 【故障恢复】
   - 启用 WAL 保证数据一致性
   - 定期备份快照文件
   - 测试恢复流程
"""
    
    print(setup_guide)


if __name__ == "__main__":
    # 运行所有演示
    demo_basic_rag()
    print("\n")
    
    demo_metadata_filtering()
    print("\n")
    
    demo_rag_pipeline()
    print("\n")
    
    demo_production_setup()
    
    print("\n✅ 所有演示完成！")
    print("\n💡 下一步:")
    print("  1. 使用真实的嵌入模型（sentence-transformers）")
    print("  2. 加载真实文档数据")
    print("  3. 启用 REST API 服务进行分布式部署")
    print("  4. 集成你的 LLM（OpenAI、Claude 等）")
