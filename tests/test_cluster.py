"""
测试分布式向量数据库集群功能

Phase 5 集群测试
"""

import pytest
from sqcvecdb.cluster.consistent_hash import ConsistentHash
from sqcvecdb.cluster.client import DistributedVectorDBClient


class TestConsistentHash:
    """一致性哈希测试"""

    def test_add_remove_nodes(self):
        """测试节点添加和移除"""
        ch = ConsistentHash()
        assert len(ch.nodes) == 0

        ch.add_node("node1")
        assert "node1" in ch.nodes

        ch.add_node("node2")
        assert "node2" in ch.nodes

        ch.remove_node("node1")
        assert "node1" not in ch.nodes
        assert "node2" in ch.nodes

    def test_key_mapping(self):
        """测试 key 映射到节点"""
        ch = ConsistentHash(["node1", "node2", "node3"])

        # 每个 key 都应该映射到某个节点
        for i in range(100):
            node = ch.get_node(f"key_{i}")
            assert node in ["node1", "node2", "node3"]

    def test_balanced_distribution(self):
        """测试分布均衡性"""
        ch = ConsistentHash(["node1", "node2", "node3"])

        distribution = {"node1": 0, "node2": 0, "node3": 0}
        for i in range(1000):
            node = ch.get_node(f"key_{i}")
            distribution[node] += 1

        # 每个节点应该大约分到 1/3 的 key（允许误差）
        for count in distribution.values():
            assert 200 < count < 400, f"Distribution {distribution} not balanced"

    def test_consistency_on_node_removal(self):
        """测试节点移除时的一致性"""
        ch = ConsistentHash(["node1", "node2", "node3"])

        # 第一阶段：获取所有 key 的节点
        before = {}
        for i in range(100):
            before[f"key_{i}"] = ch.get_node(f"key_{i}")

        # 第二阶段：移除一个节点
        ch.remove_node("node2")

        # 第三阶段：验证不受影响的 key 仍然映射到原节点
        affected_count = 0
        for key, original_node in before.items():
            new_node = ch.get_node(key)
            if original_node != "node2" and original_node != new_node:
                affected_count += 1

        # 只有约 1/3 的 key（原来映射到 node2 的）会改变
        # 其他 key 应该保持不变
        assert affected_count < 50, "Too many unaffected keys changed"

    def test_replica_nodes(self):
        """测试副本节点分配"""
        ch = ConsistentHash(["node1", "node2", "node3", "node4"])

        # 获取 key 的 2 个副本节点
        replicas = ch.get_nodes("key_1", replica_count=2)
        assert len(replicas) == 2
        assert replicas[0] != replicas[1]
        assert replicas[0] in ["node1", "node2", "node3", "node4"]
        assert replicas[1] in ["node1", "node2", "node3", "node4"]

    def test_replica_count_exceeds_nodes(self):
        """测试副本数超过节点数的情况"""
        ch = ConsistentHash(["node1", "node2"])

        replicas = ch.get_nodes("key_1", replica_count=5)
        # 应该只返回实际存在的节点数
        assert len(replicas) == 2

    def test_empty_hash_ring(self):
        """测试空哈希环"""
        ch = ConsistentHash()

        node = ch.get_node("key_1")
        assert node is None

        replicas = ch.get_nodes("key_1", replica_count=2)
        assert replicas == []


class TestDistributedVectorDBClient:
    """分布式客户端测试（基于一致性哈希）"""

    def test_client_initialization(self):
        """测试客户端初始化"""
        nodes = [
            "http://localhost:8001",
            "http://localhost:8002",
            "http://localhost:8003",
        ]
        client = DistributedVectorDBClient(nodes)

        assert len(client.nodes) == 3
        assert client.replica_count == 1

    def test_collection_selection(self):
        """测试 collection 选择"""
        nodes = ["http://localhost:8001"]
        client = DistributedVectorDBClient(nodes)

        assert client._collection_name is None
        client.use_collection("my_collection")
        assert client._collection_name == "my_collection"

    def test_primary_node_selection(self):
        """测试主节点选择"""
        nodes = [
            "http://localhost:8001",
            "http://localhost:8002",
            "http://localhost:8003",
        ]
        client = DistributedVectorDBClient(nodes)

        # 相同的 vec_id 应该总是映射到相同的节点
        for _ in range(5):
            node = client._get_primary_node(123)
            assert node in nodes
            node2 = client._get_primary_node(123)
            assert node == node2

    def test_replica_node_selection(self):
        """测试副本节点选择"""
        nodes = [
            "http://localhost:8001",
            "http://localhost:8002",
            "http://localhost:8003",
        ]
        client = DistributedVectorDBClient(nodes, replica_count=2)

        replicas = client._get_replica_nodes(123)
        assert len(replicas) == 2
        assert replicas[0] != replicas[1]

    def test_add_remove_nodes_at_runtime(self):
        """测试运行时添加/移除节点"""
        nodes = ["http://localhost:8001", "http://localhost:8002"]
        client = DistributedVectorDBClient(nodes)

        assert len(client.nodes) == 2

        # 添加节点
        client.add_node("http://localhost:8003")
        assert len(client.nodes) == 3
        assert "http://localhost:8003" in client.nodes

        # 移除节点
        client.remove_node("http://localhost:8001")
        assert len(client.nodes) == 2
        assert "http://localhost:8001" not in client.nodes

    def test_error_on_collection_not_selected(self):
        """测试未选择 collection 时的错误"""
        client = DistributedVectorDBClient(["http://localhost:8001"])

        with pytest.raises(ValueError, match="Collection not selected"):
            client.insert(1, [0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="Collection not selected"):
            client.delete(1)

        with pytest.raises(ValueError, match="Collection not selected"):
            client.search([0.1, 0.2, 0.3, 0.4])

    def test_error_on_no_available_nodes(self):
        """测试无可用节点时的错误"""
        client = DistributedVectorDBClient([])

        client.use_collection("my_collection")

        with pytest.raises(RuntimeError, match="No available nodes"):
            client._get_primary_node(1)


class TestConsistentHashIntegration:
    """一致性哈希集成测试"""

    def test_hash_ring_stability(self):
        """验证哈希环的稳定性：节点变化不影响其他 key"""
        ch = ConsistentHash(["node1", "node2", "node3"])

        # 记录初始分配
        initial_mapping = {}
        for i in range(100):
            initial_mapping[i] = ch.get_node(str(i))

        # 添加新节点
        ch.add_node("node4")

        # 验证只有部分 key 重新分配
        changed_count = 0
        for i in range(100):
            new_node = ch.get_node(str(i))
            if initial_mapping[i] != new_node:
                changed_count += 1

        # 预期变化：约 1/4 的 key（原来映射到其他位置现在映射到 node4）
        # 但要保持在合理范围（考虑虚拟节点数量）
        assert 10 < changed_count < 50, f"Changed {changed_count} keys, expected 10-50"

    def test_virtual_nodes_effect(self):
        """测试虚拟节点数量对分布的影响"""
        ch1 = ConsistentHash(["node1", "node2"], virtual_nodes=10)
        ch2 = ConsistentHash(["node1", "node2"], virtual_nodes=1000)

        dist1 = {"node1": 0, "node2": 0}
        dist2 = {"node1": 0, "node2": 0}

        for i in range(1000):
            dist1[ch1.get_node(str(i))] += 1
            dist2[ch2.get_node(str(i))] += 1

        # 虚拟节点数越多，分布应该越均匀
        variance1 = max(dist1.values()) - min(dist1.values())
        variance2 = max(dist2.values()) - min(dist2.values())

        print(f"Distribution with 10 virtual nodes: {dist1}, variance: {variance1}")
        print(f"Distribution with 1000 virtual nodes: {dist2}, variance: {variance2}")

        # 虚拟节点多的分布应该更均匀（方差更小）
        # 注意：这是统计性质，不是绝对保证
        assert variance2 <= variance1 * 1.5
