"""
一致性哈希环实现

用于分布式分片中的节点布局。支持虚拟节点以实现更均匀的分布。
"""

import hashlib
from bisect import bisect_right
from typing import List, Optional


class ConsistentHash:
    """一致性哈希环。

    特点：
    - 支持动态添加/移除节点
    - 虚拟节点机制实现均匀分布
    - 对象 key 分配到距离最近的后继节点

    用途：
    - 向量 id 到 shard 节点的映射
    - 故障时自动转移负载
    """

    def __init__(self, nodes: Optional[List[str]] = None, virtual_nodes: int = 160):
        """
        初始化一致性哈希环。

        Args:
            nodes: 初始节点列表（如 ["node1", "node2"]）
            virtual_nodes: 每个真实节点的虚拟节点数量，越大分布越均匀
        """
        self.virtual_nodes = virtual_nodes
        self.ring: dict[int, str] = {}  # hash -> node_name 映射
        self.sorted_keys: list[int] = []  # 排序的 hash 值
        self.nodes = set()  # 真实节点集合

        if nodes:
            for node in nodes:
                self.add_node(node)

    def _hash(self, key: str) -> int:
        """计算 key 的 hash 值。"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node: str) -> None:
        """添加节点。"""
        if node in self.nodes:
            return

        self.nodes.add(node)
        # 为节点创建虚拟节点
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = node
            self.sorted_keys.append(hash_value)

        self.sorted_keys.sort()

    def remove_node(self, node: str) -> None:
        """移除节点。"""
        if node not in self.nodes:
            return

        self.nodes.remove(node)
        # 移除虚拟节点
        for i in range(self.virtual_nodes):
            virtual_key = f"{node}:{i}"
            hash_value = self._hash(virtual_key)
            if hash_value in self.ring:
                del self.ring[hash_value]
            if hash_value in self.sorted_keys:
                self.sorted_keys.remove(hash_value)

        self.sorted_keys.sort()

    def get_node(self, key: str) -> Optional[str]:
        """获取 key 对应的节点。

        算法：
        1. 计算 key 的 hash 值
        2. 在环上找最近的后继节点
        3. 若环为空返回 None
        """
        if not self.ring:
            return None

        hash_value = self._hash(key)
        # 二分查找最近的后继节点位置
        idx = bisect_right(self.sorted_keys, hash_value)
        # 环形：超过末尾则回到开头
        if idx == len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def get_nodes(self, key: str, replica_count: int = 1) -> List[str]:
        """获取 key 对应的 replica_count 个副本节点（不重复）。

        用于副本分配：第一个是主节点，之后是副本节点。
        """
        if not self.ring:
            return []

        if replica_count > len(self.nodes):
            replica_count = len(self.nodes)

        nodes = []
        hash_value = self._hash(key)
        idx = bisect_right(self.sorted_keys, hash_value)

        while len(nodes) < replica_count:
            if idx == len(self.sorted_keys):
                idx = 0

            node = self.ring[self.sorted_keys[idx]]
            if node not in nodes:
                nodes.append(node)

            idx += 1

        return nodes

    def get_nodes_for_range(self, start_key: str, end_key: str) -> set[str]:
        """获取覆盖 [start_key, end_key) 范围的所有节点。

        用于范围查询。
        """
        if not self.ring:
            return set()

        start_hash = self._hash(start_key)
        end_hash = self._hash(end_key)

        covered_nodes = set()

        if start_hash <= end_hash:
            # 范围不跨越环端点
            for hash_val, node in self.ring.items():
                if start_hash <= hash_val < end_hash:
                    covered_nodes.add(node)
        else:
            # 范围跨越环端点
            for hash_val, node in self.ring.items():
                if hash_val >= start_hash or hash_val < end_hash:
                    covered_nodes.add(node)

        return covered_nodes if covered_nodes else self.nodes

    def __repr__(self) -> str:
        return f"ConsistentHash(nodes={self.nodes}, virtual_nodes={self.virtual_nodes})"
