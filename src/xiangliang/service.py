"""
xiangliang REST API 服务

支持多 collection 管理、向量查询、元数据过滤等
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Dict, List, Any
import json
from datetime import datetime

from xiangliang.collection import Collection


# ==================== Pydantic Models ====================

class CollectionSchema(BaseModel):
    """Collection 配置 Schema"""
    dim: int = Field(..., gt=0, description="向量维度")
    metric: str = Field(default="l2", description="距离度量：l2/cosine/ip")
    index_type: str = Field(default="flat", description="索引类型：flat/ivf/hnsw")
    index_params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="索引参数")
    enable_wal: bool = Field(default=False, description="是否启用 WAL")


class InsertRequest(BaseModel):
    """单条插入请求"""
    id: int
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None


class InsertManyRequest(BaseModel):
    """批量插入请求"""
    records: List[InsertRequest]


class SearchRequest(BaseModel):
    """查询请求"""
    query: List[float]
    top_k: int = Field(default=5, gt=0)
    filter: Optional[Dict[str, Any]] = None


class DeleteRequest(BaseModel):
    """删除请求"""
    id: int


class CollectionInfo(BaseModel):
    """Collection 信息响应"""
    name: str
    dim: int
    metric: str
    index_type: str
    index_params: Dict[str, Any]
    num_items: int
    enable_wal: bool
    created_at: str


class SearchResult(BaseModel):
    """查询结果"""
    id: int
    distance: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """查询响应"""
    results: List[SearchResult]
    query_time_ms: float


# ==================== FastAPI App ====================

class XianliangService:
    """xiangliang 向量数据库服务"""

    def __init__(self, data_dir: str = "./xiangliang_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collections: Dict[str, Collection] = {}
        self.collection_configs: Dict[str, Dict[str, Any]] = {}
        self._load_collections()

    def _load_collections(self) -> None:
        """启动时从磁盘加载已有的 collection"""
        config_file = self.data_dir / "collections.json"
        if config_file.exists():
            with open(config_file) as f:
                configs = json.load(f)
                for name, config in configs.items():
                    col_dir = self.data_dir / name
                    if col_dir.exists():
                        try:
                            col = Collection.load(
                                str(col_dir),
                                enable_wal=config.get("enable_wal", False),
                            )
                            self.collections[name] = col
                            self.collection_configs[name] = config
                        except Exception as e:
                            print(f"Failed to load collection '{name}': {e}")

    def _save_collection_configs(self) -> None:
        """持久化 collection 配置"""
        config_file = self.data_dir / "collections.json"
        with open(config_file, "w") as f:
            json.dump(self.collection_configs, f, indent=2)

    def create_collection(self, name: str, schema: CollectionSchema) -> CollectionInfo:
        """创建新 collection"""
        if name in self.collections:
            raise ValueError(f"Collection '{name}' already exists")

        col_dir = self.data_dir / name
        col_dir.mkdir(parents=True, exist_ok=True)

        col = Collection(
            dim=schema.dim,
            metric=schema.metric,
            index_type=schema.index_type,
            index_params=schema.index_params or {},
            data_dir=str(col_dir),
            enable_wal=schema.enable_wal,
        )
        self.collections[name] = col
        self.collection_configs[name] = {
            "dim": schema.dim,
            "metric": schema.metric,
            "index_type": schema.index_type,
            "index_params": schema.index_params or {},
            "enable_wal": schema.enable_wal,
            "created_at": datetime.now().isoformat(),
        }
        self._save_collection_configs()

        return self._get_collection_info(name)

    def delete_collection(self, name: str) -> None:
        """删除 collection（从内存和磁盘）"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found")

        col = self.collections.pop(name)
        col.close()

        self.collection_configs.pop(name, None)
        self._save_collection_configs()

        # 删除磁盘目录
        col_dir = self.data_dir / name
        import shutil

        shutil.rmtree(col_dir, ignore_errors=True)

    def list_collections(self) -> List[CollectionInfo]:
        """列出所有 collection"""
        return [self._get_collection_info(name) for name in self.collections.keys()]

    def _get_collection_info(self, name: str) -> CollectionInfo:
        """获取 collection 信息"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found")

        col = self.collections[name]
        config = self.collection_configs.get(name, {})

        return CollectionInfo(
            name=name,
            dim=col.dim,
            metric=col.metric,
            index_type=col.index_type,
            index_params=col.index_params,
            num_items=len(col),
            enable_wal=col.enable_wal,
            created_at=config.get("created_at", ""),
        )

    def insert_one(self, name: str, req: InsertRequest) -> None:
        """插入单条向量"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found")

        col = self.collections[name]
        col.insert(req.id, req.vector, metadata=req.metadata)

    def insert_many(self, name: str, req: InsertManyRequest) -> int:
        """批量插入向量，返回插入数量"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found")

        col = self.collections[name]
        vectors = []
        ids = []
        metadata_list = []

        for record in req.records:
            ids.append(record.id)
            vectors.append(record.vector)
            metadata_list.append(record.metadata or {})

        col.insert_many(ids, vectors, metadata_list)
        return len(req.records)

    def delete(self, name: str, req: DeleteRequest) -> None:
        """删除向量"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found")

        col = self.collections[name]
        col.delete([req.id])  # delete 期望一个列表

    def search(self, name: str, req: SearchRequest) -> SearchResponse:
        """查询向量"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found")

        import time

        col = self.collections[name]
        start = time.time()

        results = col.search(
            req.query, top_k=req.top_k, filter=req.filter
        )

        elapsed_ms = (time.time() - start) * 1000

        search_results = [
            SearchResult(
                id=r[0],
                distance=r[1],
                metadata=col._metadata.get(r[0])  # 从 collection 的元数据存储中获取
            )
            for r in results
        ]

        return SearchResponse(results=search_results, query_time_ms=elapsed_ms)

    def checkpoint(self, name: str) -> None:
        """手动触发 collection 的 checkpoint"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found")

        col = self.collections[name]
        if not col.enable_wal:
            raise ValueError(f"Collection '{name}' has WAL disabled")

        col.checkpoint()

    def get_stats(self, name: str) -> Dict[str, Any]:
        """获取 collection 的统计信息"""
        if name not in self.collections:
            raise ValueError(f"Collection '{name}' not found")

        col = self.collections[name]
        return {
            "name": name,
            "dim": col.dim,
            "metric": col.metric,
            "index_type": col.index_type,
            "num_items": len(col),
            "enable_wal": col.enable_wal,
        }

    def close_all(self) -> None:
        """关闭所有 collection"""
        for col in self.collections.values():
            col.close()


# ==================== FastAPI Routes ====================

app = FastAPI(
    title="xiangliang Vector DB API",
    description="开源向量数据库 REST API",
    version="0.2.0",
)

# 全局服务实例
service: Optional[XianliangService] = None


def init_service(data_dir: str = "./xiangliang_data") -> None:
    """初始化服务（在 app startup 时调用）"""
    global service
    service = XianliangService(data_dir)


@app.on_event("startup")
async def startup():
    """应用启动时初始化"""
    init_service()


@app.on_event("shutdown")
async def shutdown():
    """应用关闭时清理"""
    if service:
        service.close_all()


@app.get("/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


# ========== Collection 管理 ==========


@app.post("/collections", response_model=CollectionInfo)
async def create_collection(name: str, schema: CollectionSchema):
    """创建新 collection"""
    try:
        return service.create_collection(name, schema)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/collections", response_model=List[CollectionInfo])
async def list_collections():
    """列出所有 collection"""
    return service.list_collections()


@app.get("/collections/{name}", response_model=CollectionInfo)
async def get_collection(name: str):
    """获取 collection 信息"""
    try:
        return service._get_collection_info(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/collections/{name}")
async def delete_collection(name: str):
    """删除 collection"""
    try:
        service.delete_collection(name)
        return {"message": f"Collection '{name}' deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/collections/{name}/stats")
async def get_stats(name: str):
    """获取 collection 统计信息"""
    try:
        return service.get_stats(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ========== 向量操作 ==========


@app.post("/collections/{name}/insert")
async def insert_one(name: str, req: InsertRequest):
    """插入单条向量"""
    try:
        service.insert_one(name, req)
        return {"message": "Vector inserted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/collections/{name}/insert_many")
async def insert_many(name: str, req: InsertManyRequest):
    """批量插入向量"""
    try:
        count = service.insert_many(name, req)
        return {"message": f"{count} vectors inserted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/collections/{name}/delete")
async def delete(name: str, req: DeleteRequest):
    """删除向量"""
    try:
        service.delete(name, req)
        return {"message": "Vector deleted"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/collections/{name}/search", response_model=SearchResponse)
async def search(name: str, req: SearchRequest):
    """查询向量"""
    try:
        return service.search(name, req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ========== WAL 操作 ==========


@app.post("/collections/{name}/checkpoint")
async def checkpoint(name: str):
    """手动触发 checkpoint"""
    try:
        service.checkpoint(name)
        return {"message": "Checkpoint completed"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
