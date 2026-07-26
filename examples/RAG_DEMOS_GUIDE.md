# SQCVecDB RAG 演示指南

本目录包含三个完整的 RAG（检索增强生成）系统演示，展示如何使用 SQCVecDB 框架构建现实应用。

## 📚 三个演示层次

### 1. **rag_quickstart.py** - 快速入门（⭐ 推荐新手）

**适合人群**：刚接触 RAG 的开发者

**核心功能**：
- 简化的 SimpleRAG 类，5 分钟上手
- 内置简单向量化工具
- 支持基础文档管理和检索
- 生成 LLM 提示词模板

**特点**：
```python
# 3 行代码启动 RAG 系统
rag = SimpleRAG(dim=256, index_type="ivf")
rag.add_document("标题", "内容")
prompt = rag.generate_prompt("用户查询")
```

**运行方式**：
```bash
python examples/rag_quickstart.py
```

**适用场景**：
- 学习 RAG 基础概念
- 原型设计
- 小规模文档库

---

### 2. **rag_system_demo.py** - 完整系统（⭐⭐ 推荐学习）

**适合人群**：想深入理解 RAG 架构的开发者

**核心功能**：
- SimpleEmbedder：可自定义的向量化
- DocumentChunker：智能文本分块
- SimpleRAGSystem：生产级 RAG 系统
- 支持 WAL 持久化
- 完整的系统工作流程

**特点**：
```python
# 完整的 RAG 系统
rag = SimpleRAGSystem(dim=384, index_type="hnsw")
rag.ingest_documents(documents)  # 批量导入
result = rag.query(query, top_k=3)  # 完整流程
```

**四个演示模块**：
1. 基础 RAG - 文档检索和生成
2. 带过滤的 RAG - 按类别检索
3. 索引类型对比 - Flat vs IVF vs HNSW
4. 持久化和加载 - 数据保存恢复

**运行方式**：
```bash
python examples/rag_system_demo.py
```

**适用场景**：
- 学习 RAG 完整架构
- 中等规模知识库
- 性能对比测试

---

### 3. **rag_with_llm.py** - 生产集成（⭐⭐⭐ 推荐生产）

**适合人群**：构建实际生产应用的工程师

**核心功能**：
- RAGComponentRegistry：可插拔组件注册表
- 支持多种嵌入器（默认、OpenAI）
- 支持多种 LLM（本地、OpenAI、Claude）
- 灵活的过滤检索
- 来源溯源功能

**特点**：
```python
# 可插拔组件设计
registry.register_embedder("custom", your_embedder)
registry.register_llm("gpt4", your_llm)

rag = AdvancedRAG(
    dim=384,
    embedder_name="custom",
    llm_name="gpt4"
)
```

**核心特性**：
1. **组件可插拔** - 轻松集成不同的嵌入模型和 LLM
2. **灵活过滤** - 按类别、来源等多条件过滤
3. **完整流程** - 从检索到生成的整个 RAG 流程
4. **生产级** - 元数据管理、错误处理、性能优化

**运行方式**：
```bash
python examples/rag_with_llm.py
```

**扩展示例**：
```python
# 集成 Sentence Transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
registry.register_embedder("st", lambda text, dim: model.encode(text))

# 集成 OpenAI
import os
from openai import OpenAI
openai_llm = openai_llm_factory(os.getenv("OPENAI_API_KEY"))
registry.register_llm("gpt-4", openai_llm)
```

**适用场景**：
- 生产环境部署
- 大规模知识库
- 多模型集成
- 企业级应用

---

## 🚀 快速对比

| 特性 | quickstart | system_demo | with_llm |
|------|-----------|------------|----------|
| **复杂度** | ⭐ 简单 | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 高级 |
| **代码量** | ~250 行 | ~600 行 | ~450 行 |
| **学习难度** | 容易 | 中等 | 中等-困难 |
| **生产就绪** | ✅ | ✅✅ | ✅✅✅ |
| **向量化** | 内置简单版 | 自定义版本 | 可插拔 |
| **LLM 集成** | 模拟版本 | 模拟版本 | 完全集成 |
| **持久化** | ❌ | ✅ | ✅ |
| **过滤检索** | ❌ | ✅ | ✅ |
| **索引类型** | 多种 | 多种 | 多种 |
| **元数据** | 基础 | 完善 | 完善 |

---

## 📖 学习路径

### 初学者路径
1. **阅读 README.md** - 了解项目整体
2. **运行 rag_quickstart.py** - 了解基础概念
3. **修改 quickstart.py** - 练习基本 API
4. **阅读源代码** - 理解底层实现

### 进阶开发者路径
1. **运行 rag_system_demo.py** - 理解完整架构
2. **分析不同索引性能** - 选择合适的索引
3. **学习文档分块策略** - 优化检索效果
4. **集成持久化** - 管理大规模数据

### 生产工程师路径
1. **运行 rag_with_llm.py** - 了解生产设计
2. **集成真实嵌入模型** - 使用 sentence-transformers 或 OpenAI
3. **集成真实 LLM** - 使用 OpenAI/Claude/本地模型
4. **性能优化和扩展** - 异步处理、缓存等

---

## 🛠️ 实用技巧

### 1. 选择合适的向量维度
```python
# 小规模测试：64-128 维
rag = SimpleRAG(dim=128)

# 中等规模：256-384 维
rag = SimpleRAG(dim=384)

# 大规模、高精度：768-1536 维
rag = SimpleRAG(dim=1536)
```

### 2. 选择合适的索引类型
```python
# 小数据集（<1M）：flat
rag = SimpleRAG(index_type="flat")

# 中等数据集（1M-100M）：ivf
rag = SimpleRAG(index_type="ivf")

# 大数据集（>100M）：hnsw
rag = SimpleRAG(index_type="hnsw")
```

### 3. 优化检索质量
```python
# 增加返回结果数，然后重排
results = rag.retrieve(query, top_k=10)

# 使用多条件过滤
results = rag.retrieve(query, top_k=5, category="ai")

# 调整相似度阈值
filtered = [r for r in results if r['similarity'] > 0.5]
```

### 4. 提示词工程
```python
# 为 LLM 准备高质量上下文
prompt = f"""根据以下背景信息回答问题。

【背景信息】
{context}

【问题】
{user_query}

【要求】
- 基于背景信息
- 保持准确性
- 标注来源
"""
```

---

## 💡 常见问题

**Q: 应该从哪个演示开始？**
A: 如果是新手，从 `rag_quickstart.py` 开始。如果需要学习完整架构，选择 `rag_system_demo.py`。如果要构建生产应用，使用 `rag_with_llm.py`。

**Q: 如何集成自己的嵌入模型？**
A: 在 `rag_with_llm.py` 中注册：
```python
registry.register_embedder("my_model", lambda text, dim: my_embed_fn(text))
```

**Q: 如何集成真实 LLM？**
A: 注册 LLM 生成函数：
```python
registry.register_llm("my_llm", my_llm_generator_fn)
```

**Q: 性能不好怎么办？**
A: 
- 增加 `top_k` 获取更多候选结果
- 尝试不同的索引类型（IVF、HNSW）
- 优化文档分块大小
- 使用更好的嵌入模型

**Q: 如何处理大规模文档？**
A:
- 使用 `ingest_batch()` 批量导入
- 定期 checkpoint 进行持久化
- 使用 IVF 或 HNSW 索引
- 考虑分布式部署

---

## 📚 相关资源

- [SQCVecDB 官方文档](../README.md)
- [向量数据库基础](../README.md#架构设计)
- [索引算法详解](../src/xiangliang/index/)
- [测试用例参考](../tests/)

---

## 🤝 贡献

欢迎提交问题、改进建议或新的演示示例！

---

## 📄 许可证

同 SQCVecDB 项目

---

**祝你使用 SQCVecDB 构建出色的 RAG 应用！** 🚀
