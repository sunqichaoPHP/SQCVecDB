# 安装指南

SQCVecDB 支持多种安装方式，选择最适合你的方式进行安装。

## 系统要求

- **Python**: 3.9 及以上
- **操作系统**: Linux、macOS、Windows
- **内存**: 最少 2GB（推荐 4GB+）

## 1. pip 安装（推荐）

### 基础安装

```bash
pip install SQCVecDB
```

### 含服务依赖安装（REST API）

```bash
pip install SQCVecDB[service]
```

### 含分布式依赖安装（集群模式）

```bash
pip install SQCVecDB[cluster]
```

### 完整安装（包含所有可选依赖）

```bash
pip install SQCVecDB[dev,service,cluster]
```

## 2. 从源码安装

### 克隆仓库

```bash
git clone https://github.com/yourusername/SQCVecDB.git
cd SQCVecDB
```

### 创建虚拟环境

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 安装开发版本

```bash
# 基础安装
pip install -e .

# 包含所有依赖
pip install -e ".[dev,service,cluster]"
```

## 3. 依赖说明

### 核心依赖
- `numpy>=1.24` — 向量计算

### 可选依赖

| 分组 | 目的 | 包含 |
|------|------|------|
| `service` | REST API 服务 | fastapi, uvicorn, pydantic |
| `cluster` | 分布式集群 | requests |
| `dev` | 开发工具 | pytest, ruff, httpx |

## 4. 验证安装

### 快速测试

```python
from xiangliang.collection import Collection
import numpy as np

# 创建 Collection
col = Collection(name="test", dim=10)

# 插入向量
col.insert("1", np.random.rand(10))

# 搜索
results = col.search(np.random.rand(10), top_k=1)
print("✅ 安装成功！")
```

### 运行测试

```bash
# 安装测试依赖
pip install pytest

# 运行测试
pytest tests/ -v
```

## 5. 快速开始

### Python SDK 使用

```python
from xiangliang.collection import Collection

# 创建向量数据库
col = Collection(name="my_db", dim=384, index_type="ivf")

# 插入向量
col.insert(
    id="doc_1",
    vector=[0.1, 0.2, ..., 0.384],
    metadata={"source": "doc.md"}
)

# 搜索
results = col.search([0.1, 0.2, ..., 0.384], top_k=5)

# 持久化
col.checkpoint()
```

### REST API 使用

```bash
# 安装服务依赖
pip install SQCVecDB[service]

# 启动服务
uvicorn xiangliang.service:app --port 8000

# 在另一个终端测试
curl -X POST http://localhost:8000/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "test", "dim": 10, "index_type": "flat"}'
```

## 6. 常见问题

### Q: 如何更新 SQCVecDB？

```bash
pip install --upgrade SQCVecDB
```

### Q: 如何卸载 SQCVecDB？

```bash
pip uninstall SQCVecDB
```

### Q: 需要 GPU 支持吗？

目前不需要，所有操作都在 CPU 上。未来版本可能支持 GPU 加速。

### Q: 支持哪些 Python 版本？

支持 Python 3.9+。推荐使用最新的 LTS 版本（3.10 或 3.11）。

### Q: 如何在 Windows 上安装？

步骤相同，只需在命令行中用 `.venv\Scripts\activate` 替代 `source .venv/bin/activate`。

### Q: 如何处理安装错误？

常见解决方案：

```bash
# 升级 pip
pip install --upgrade pip

# 清除缓存
pip cache purge

# 重新安装
pip install --no-cache-dir SQCVecDB[service]
```

## 7. Docker 容器（可选）

如果你想在 Docker 中使用 SQCVecDB，可以创建 Dockerfile：

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
RUN pip install --no-cache-dir SQCVecDB[service]

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "xiangliang.service:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建并运行：

```bash
docker build -t sqcvecdb .
docker run -p 8000:8000 sqcvecdb
```

## 8. 开发环境设置

如果你想为项目贡献代码：

```bash
# 克隆仓库
git clone https://github.com/yourusername/SQCVecDB.git
cd SQCVecDB

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS

# 安装所有依赖
pip install -e ".[dev,service,cluster]"

# 验证安装
pytest -q
```

## 9. 后续步骤

- 📖 阅读 [README.md](README.md) 了解更多特性
- 🔧 查看 [examples/](examples/) 中的示例代码
- 🤝 如有问题，提交 [Issue](https://github.com/yourusername/SQCVecDB/issues)

---

**需要帮助？** 查看 [CONTRIBUTING.md](CONTRIBUTING.md) 或访问 [GitHub Discussions](https://github.com/yourusername/SQCVecDB/discussions)
