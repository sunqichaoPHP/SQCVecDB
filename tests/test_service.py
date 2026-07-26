"""
测试 xiangliang REST API 服务
"""

import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from xiangliang.service import app, init_service


@pytest.fixture
def temp_data_dir():
    """临时数据目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def client(temp_data_dir):
    """FastAPI 测试客户端"""
    init_service(temp_data_dir)
    return TestClient(app)


class TestHealth:
    """健康检查测试"""

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestCollectionManagement:
    """Collection 管理测试"""

    def test_create_collection(self, client):
        response = client.post(
            "/collections",
            params={"name": "test_col"},
            json={
                "dim": 4,
                "metric": "l2",
                "index_type": "flat",
                "enable_wal": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_col"
        assert data["dim"] == 4
        assert data["num_items"] == 0

    def test_list_collections(self, client):
        # 创建两个 collection
        client.post(
            "/collections",
            params={"name": "col1"},
            json={"dim": 4, "metric": "l2", "index_type": "flat"},
        )
        client.post(
            "/collections",
            params={"name": "col2"},
            json={"dim": 8, "metric": "cosine", "index_type": "ivf"},
        )

        response = client.get("/collections")
        assert response.status_code == 200
        collections = response.json()
        assert len(collections) == 2
        names = {c["name"] for c in collections}
        assert names == {"col1", "col2"}

    def test_get_collection(self, client):
        client.post(
            "/collections",
            params={"name": "test_col"},
            json={"dim": 4, "metric": "l2", "index_type": "flat"},
        )

        response = client.get("/collections/test_col")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_col"

    def test_get_nonexistent_collection(self, client):
        response = client.get("/collections/nonexistent")
        assert response.status_code == 404

    def test_create_duplicate_collection(self, client):
        client.post(
            "/collections",
            params={"name": "test_col"},
            json={"dim": 4, "metric": "l2", "index_type": "flat"},
        )

        response = client.post(
            "/collections",
            params={"name": "test_col"},
            json={"dim": 4, "metric": "l2", "index_type": "flat"},
        )
        assert response.status_code == 400

    def test_delete_collection(self, client):
        client.post(
            "/collections",
            params={"name": "test_col"},
            json={"dim": 4, "metric": "l2", "index_type": "flat"},
        )

        response = client.delete("/collections/test_col")
        assert response.status_code == 200

        response = client.get("/collections/test_col")
        assert response.status_code == 404


class TestVectorOperations:
    """向量操作测试"""

    @pytest.fixture
    def collection(self, client):
        """创建测试用 collection"""
        client.post(
            "/collections",
            params={"name": "test_col"},
            json={"dim": 4, "metric": "l2", "index_type": "flat"},
        )
        return "test_col"

    def test_insert_one(self, client, collection):
        response = client.post(
            f"/collections/{collection}/insert",
            json={"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "metadata": {"name": "item1"}},
        )
        assert response.status_code == 200

        # 验证插入
        stats = client.get(f"/collections/{collection}/stats")
        assert stats.json()["num_items"] == 1

    def test_insert_many(self, client, collection):
        response = client.post(
            f"/collections/{collection}/insert_many",
            json={
                "records": [
                    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
                    {"id": 2, "vector": [0.5, 0.6, 0.7, 0.8]},
                    {"id": 3, "vector": [0.9, 0.8, 0.7, 0.6]},
                ]
            },
        )
        assert response.status_code == 200
        assert "3 vectors inserted" in response.json()["message"]

        stats = client.get(f"/collections/{collection}/stats")
        assert stats.json()["num_items"] == 3

    def test_search(self, client, collection):
        # 插入数据
        client.post(
            f"/collections/{collection}/insert_many",
            json={
                "records": [
                    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
                    {"id": 2, "vector": [0.5, 0.6, 0.7, 0.8]},
                    {"id": 3, "vector": [0.9, 0.8, 0.7, 0.6]},
                ]
            },
        )

        # 查询
        response = client.post(
            f"/collections/{collection}/search",
            json={"query": [0.1, 0.2, 0.3, 0.4], "top_k": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        assert data["results"][0]["id"] == 1  # 最接近的

    def test_delete(self, client, collection):
        # 插入数据
        client.post(
            f"/collections/{collection}/insert",
            json={"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
        )

        # 删除
        response = client.post(
            f"/collections/{collection}/delete",
            json={"id": 1},
        )
        assert response.status_code == 200

        # 验证删除
        stats = client.get(f"/collections/{collection}/stats")
        assert stats.json()["num_items"] == 0


class TestWALOperations:
    """WAL 操作测试"""

    @pytest.fixture
    def wal_collection(self, client):
        """创建启用 WAL 的 collection"""
        client.post(
            "/collections",
            params={"name": "wal_col"},
            json={
                "dim": 4,
                "metric": "l2",
                "index_type": "flat",
                "enable_wal": True,
            },
        )
        return "wal_col"

    def test_checkpoint(self, client, wal_collection):
        # 插入数据
        client.post(
            f"/collections/{wal_collection}/insert_many",
            json={
                "records": [
                    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4]},
                    {"id": 2, "vector": [0.5, 0.6, 0.7, 0.8]},
                ]
            },
        )

        # 触发 checkpoint
        response = client.post(f"/collections/{wal_collection}/checkpoint")
        assert response.status_code == 200
        assert "Checkpoint completed" in response.json()["message"]

    def test_checkpoint_without_wal(self, client):
        # 创建不启用 WAL 的 collection
        client.post(
            "/collections",
            params={"name": "no_wal_col"},
            json={"dim": 4, "metric": "l2", "index_type": "flat", "enable_wal": False},
        )

        # 尝试 checkpoint 应该失败
        response = client.post("/collections/no_wal_col/checkpoint")
        assert response.status_code == 404


class TestIndexTypes:
    """不同索引类型测试"""

    def test_flat_index(self, client):
        client.post(
            "/collections",
            params={"name": "flat_col"},
            json={"dim": 4, "metric": "l2", "index_type": "flat"},
        )

        info = client.get("/collections/flat_col").json()
        assert info["index_type"] == "flat"

    def test_ivf_index(self, client):
        client.post(
            "/collections",
            params={"name": "ivf_col"},
            json={
                "dim": 4,
                "metric": "l2",
                "index_type": "ivf",
                "index_params": {"nlist": 10, "nprobe": 3},
            },
        )

        info = client.get("/collections/ivf_col").json()
        assert info["index_type"] == "ivf"
        assert info["index_params"]["nlist"] == 10

    def test_hnsw_index(self, client):
        client.post(
            "/collections",
            params={"name": "hnsw_col"},
            json={
                "dim": 4,
                "metric": "l2",
                "index_type": "hnsw",
                "index_params": {"M": 16, "ef_search": 64},
            },
        )

        info = client.get("/collections/hnsw_col").json()
        assert info["index_type"] == "hnsw"
        assert info["index_params"]["M"] == 16


class TestMetadataFiltering:
    """元数据过滤测试"""

    @pytest.fixture
    def collection_with_metadata(self, client):
        """创建带元数据的 collection"""
        client.post(
            "/collections",
            params={"name": "meta_col"},
            json={"dim": 4, "metric": "l2", "index_type": "flat"},
        )

        # 插入带元数据的向量
        client.post(
            f"/collections/meta_col/insert_many",
            json={
                "records": [
                    {"id": 1, "vector": [0.1, 0.2, 0.3, 0.4], "metadata": {"category": "A"}},
                    {"id": 2, "vector": [0.15, 0.25, 0.35, 0.45], "metadata": {"category": "B"}},
                    {"id": 3, "vector": [0.9, 0.8, 0.7, 0.6], "metadata": {"category": "A"}},
                ]
            },
        )
        return "meta_col"

    def test_search_with_filter(self, client, collection_with_metadata):
        response = client.post(
            f"/collections/{collection_with_metadata}/search",
            json={
                "query": [0.1, 0.2, 0.3, 0.4],
                "top_k": 3,
                "filter": {"category": "A"},
            },
        )
        assert response.status_code == 200
        results = response.json()["results"]
        # 所有结果的 metadata 应该都满足 filter
        assert all(r["metadata"]["category"] == "A" for r in results)
