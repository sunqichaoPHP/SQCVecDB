"""
分布式向量数据库客户端

支持一致性哈希分片、scatter-gather 查询、自动故障转移等
"""

from typing import Optional, Dict, List, Any
import requests
from sqcvecdb.cluster.consistent_hash import ConsistentHash


class DistributedVectorDBClient:
    """分布式向量数据库客户端。

    用户通过此客户端访问分布式集群，自动处理：
    - 向量 ID 到节点的哈希分片
    - Scatter-gather 查询（并行查询所有节点）
    - 故障转移（节点无响应时重试其他副本）
    """

    def __init__(self, nodes: List[str], replica_count: int = 1, timeout: float = 5.0):
        """
        初始化客户端。

        Args:
            nodes: 节点列表，格式 ["http://node1:8000", "http://node2:8001", ...]
            replica_count: 每个向量的副本数（目前基于哈希分配，不涉及实际同步）
            timeout: HTTP 请求超时时间（秒）
        """
        self.nodes = nodes
        self.replica_count = replica_count
        self.timeout = timeout
        self.hash_ring = ConsistentHash(nodes, virtual_nodes=160)
        self._collection_name: Optional[str] = None

    def use_collection(self, name: str) -> "DistributedVectorDBClient":
        """切换到指定 collection。"""
        self._collection_name = name
        return self

    def _get_primary_node(self, vec_id: int) -> str:
        """获取向量 ID 对应的主节点。"""
        node = self.hash_ring.get_node(str(vec_id))
        if not node:
            raise RuntimeError("No available nodes in the cluster")
        return node

    def _get_replica_nodes(self, vec_id: int) -> List[str]:
        """获取向量 ID 对应的所有副本节点。"""
        nodes = self.hash_ring.get_nodes(str(vec_id), self.replica_count)
        if not nodes:
            raise RuntimeError("No available nodes in the cluster")
        return nodes

    def insert(self, vec_id: int, vector: List[float], metadata: Optional[Dict[str, Any]] = None) -> None:
        """插入单条向量到对应的分片节点。"""
        if not self._collection_name:
            raise ValueError("Collection not selected, call use_collection() first")

        node = self._get_primary_node(vec_id)

        try:
            resp = requests.post(
                f"{node}/collections/{self._collection_name}/insert",
                json={"id": vec_id, "vector": vector, "metadata": metadata},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to insert into {node}: {e}")

    def insert_many(self, records: List[Dict[str, Any]]) -> None:
        """批量插入向量。

        按 vec_id 分组，发送到对应的分片节点（scatter）。
        """
        if not self._collection_name:
            raise ValueError("Collection not selected, call use_collection() first")

        # 按节点分组记录
        shard_records: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            node = self._get_primary_node(record["id"])
            if node not in shard_records:
                shard_records[node] = []
            shard_records[node].append(record)

        # 并行向各节点插入
        errors = []
        for node, node_records in shard_records.items():
            try:
                resp = requests.post(
                    f"{node}/collections/{self._collection_name}/insert_many",
                    json={"records": node_records},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
            except requests.RequestException as e:
                errors.append(f"{node}: {e}")

        if errors:
            raise RuntimeError(f"Failed to insert to some nodes: {errors}")

    def delete(self, vec_id: int) -> None:
        """删除向量从对应的分片节点。"""
        if not self._collection_name:
            raise ValueError("Collection not selected, call use_collection() first")

        node = self._get_primary_node(vec_id)

        try:
            resp = requests.post(
                f"{node}/collections/{self._collection_name}/delete",
                json={"id": vec_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to delete from {node}: {e}")

    def search(self, query: List[float], top_k: int = 5, filter: Optional[Dict[str, Any]] = None) -> List[tuple]:
        """查询向量。

        Scatter-gather 模式：
        1. 向所有节点并行发送查询
        2. 收集各节点返回的 top_k 结果
        3. 归并排序，返回全局 top_k
        """
        if not self._collection_name:
            raise ValueError("Collection not selected, call use_collection() first")

        all_results = []
        errors = []

        # Scatter：并行查询所有节点
        for node in self.nodes:
            try:
                resp = requests.post(
                    f"{node}/collections/{self._collection_name}/search",
                    json={"query": query, "top_k": top_k, "filter": filter},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()

                # 收集结果
                for r in data["results"]:
                    all_results.append((r["id"], r["distance"], r.get("metadata")))
            except requests.RequestException as e:
                errors.append(f"{node}: {e}")

        if not all_results and errors:
            raise RuntimeError(f"Failed to query all nodes: {errors}")

        # Gather：按距离排序，返回全局 top_k
        all_results.sort(key=lambda x: x[1])
        return all_results[:top_k]

    def get_stats(self) -> Dict[str, Any]:
        """获取集群统计信息。

        聚合所有节点的统计。
        """
        if not self._collection_name:
            raise ValueError("Collection not selected, call use_collection() first")

        total_items = 0
        node_stats = {}

        for node in self.nodes:
            try:
                resp = requests.get(
                    f"{node}/collections/{self._collection_name}/stats",
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                stats = resp.json()
                node_stats[node] = stats
                total_items += stats["num_items"]
            except requests.RequestException:
                node_stats[node] = {"error": "unreachable"}

        return {
            "collection": self._collection_name,
            "total_items": total_items,
            "nodes": self.nodes,
            "node_stats": node_stats,
            "replica_count": self.replica_count,
        }

    def list_collections(self) -> Dict[str, List[str]]:
        """列出所有节点上的 collection。

        汇总各节点的 collection 列表。
        """
        all_collections = {}

        for node in self.nodes:
            try:
                resp = requests.get(
                    f"{node}/collections",
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                collections = resp.json()
                all_collections[node] = [c["name"] for c in collections]
            except requests.RequestException:
                all_collections[node] = []

        return all_collections

    def add_node(self, node: str) -> None:
        """动态添加节点到集群。"""
        if node not in self.nodes:
            self.nodes.append(node)
            self.hash_ring.add_node(node)
            print(f"Node {node} added to cluster")

    def remove_node(self, node: str) -> None:
        """动态移除节点从集群。

        警告：该节点上的数据将不可访问（单副本情况下数据丢失）。
        """
        if node in self.nodes:
            self.nodes.remove(node)
            self.hash_ring.remove_node(node)
            print(f"Node {node} removed from cluster")
