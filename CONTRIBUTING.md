# 贡献指南

感谢你对 SQCVecDB 的兴趣！我们欢迎任何形式的贡献。

## 🐛 报告问题

如果你发现了 bug，请在 [GitHub Issues](https://github.com/yourusername/SQCVecDB/issues) 中创建一个新的 Issue。

**提交 Issue 时，请包括**：
- 清晰的问题描述
- 复现步骤（如适用）
- 预期行为 vs 实际行为
- Python 版本、操作系统
- 错误堆栈跟踪（如适用）

## 💡 建议功能

你可以通过 [GitHub Discussions](https://github.com/yourusername/SQCVecDB/discussions) 讨论新功能建议。

## 🔧 提交代码

### 开发环境设置

```bash
# 克隆你 Fork 的仓库
git clone https://github.com/your-username/SQCVecDB.git
cd SQCVecDB

# 创建虚拟环境
python3.9+ -m venv .venv
source .venv/bin/activate  # Linux/Mac

# 安装开发依赖
pip install -e ".[dev,service,cluster]"
```

### 代码规范

1. **Python 风格**：遵循 [PEP 8](https://pep8.org/)
2. **类型提示**：所有函数必须包含类型注解
3. **文档字符串**：使用 Google 风格的文档字符串
4. **导入排序**：标准库 → 第三方 → 本地

示例：
```python
def search(
    self,
    vector: np.ndarray,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> List[Tuple[str, float, Dict]]:
    """搜索相似向量。
    
    Args:
        vector: 查询向量，形状为 (dim,)
        top_k: 返回的最相似向量数量，默认 5
        filters: 元数据过滤条件，可选
        
    Returns:
        包含 (id, distance, metadata) 的列表
    """
    pass
```

### 提交流程

1. **创建特性分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **开发并测试**
   ```bash
   # 运行测试
   pytest -v
   
   # 代码检查
   ruff check src/
   ```

3. **提交代码**
   ```bash
   git add .
   git commit -m "feat: 简洁的功能描述"
   ```

4. **推送并创建 Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request 要求

- ✅ 所有测试通过
- ✅ 代码符合风格规范
- ✅ 包含必要的类型注解
- ✅ 添加或更新相关文档
- ✅ 新功能包含相应测试

### 提交信息规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```
feat: 新功能
fix: 修复 bug
docs: 文档更新
test: 测试相关
refactor: 重构代码
perf: 性能优化
chore: 依赖更新、配置等
```

示例：
```
feat: 添加向量量化压缩支持
fix: 修复 HNSW 索引的内存泄漏问题
test: 增加分布式故障转移测试
docs: 补充 RAG 集成指南
```

## 🧪 测试

### 编写测试

```python
# tests/test_my_feature.py
import pytest
from xiangliang.collection import Collection

class TestMyFeature:
    def test_basic_functionality(self):
        col = Collection(name="test", dim=10, index_type="flat")
        col.insert("1", [0.1] * 10, metadata={"source": "test"})
        assert col.count() == 1
        
    def test_edge_case(self):
        col = Collection(name="test", dim=10)
        # 空搜索应该返回空列表
        results = col.search([0.1] * 10, top_k=5)
        assert len(results) == 0
```

### 运行测试

```bash
# 所有测试
pytest -v

# 特定文件
pytest tests/test_my_feature.py -v

# 特定测试
pytest tests/test_my_feature.py::TestMyFeature::test_basic_functionality -v

# 测试覆盖率
pytest --cov=src/xiangliang tests/
```

## 📚 文档

### 更新文档

- 编辑 `README.md` 中的使用示例
- 更新 `CHANGELOG.md` 记录版本变化
- 在 `examples/` 中添加演示脚本

### 文档风格

- 使用清晰、简洁的语言
- 提供代码示例
- 包含链接到相关文档

## 🎯 优先级

我们优先考虑以下类型的贡献：

1. 🔴 **高优先级**
   - Bug 修复
   - 性能优化
   - 安全相关

2. 🟡 **中优先级**
   - 新功能（符合项目范围）
   - 测试覆盖率提升
   - 文档改进

3. 🟢 **低优先级**
   - 代码风格调整
   - 依赖更新
   - 示例代码

## ❓ 有问题？

- 💬 参加 [GitHub Discussions](https://github.com/yourusername/SQCVecDB/discussions)
- 📧 邮件：contact@example.com

---

感谢你的贡献！🎉
