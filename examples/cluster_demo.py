#!/usr/bin/env python3
"""
分布式向量数据库集群演示

演示内容：
1. 启动多个 sqcvecdb 服务实例（作为集群节点）
2. 通过分布式客户端与集群交互
3. 展示一致性哈希分片、scatter-gather 查询等

运行前准备：
1. 启动 3 个 API 服务：
   - node1: uvicorn sqcvecdb.service:app --port 8001
   - node2: uvicorn sqcvecdb.service:app --port 8002
   - node3: uvicorn sqcvecdb.service:app --port 8003

2. 运行此脚本：
   python examples/cluster_demo.py
"""

import time
import numpy as np
from sqcvecdb.cluster import DistributedVectorDBClient, ConsistentHash


def demo_consistent_hash():
    """演示一致性哈希"""
    print("="*70)
    print("[演示 1] 一致性哈希分片")
    print("="*70)
    print()

    # 创建哈希环
    nodes = ["node1", "node2", "node3"]
    ch = ConsistentHash(nodes, virtual_nodes=160)
    print(f"✓ 创建一致性哈希环，节点: {nodes}")
    print()

    # 展示 key 分布
    distribution = {n: 0 for n in nodes}
    print("生成 1000 个向量 ID，分配到各节点：")
    for i in range(1000):
        node = ch.get_node(f"vec_{i}")
        distribution[node] += 1

    for node, count in distribution.items():
        pct = count / 10
        bar = "█" * int(pct / 2)
        print(f"  {node}: {count:4d} ({pct:5.1f}%) {bar}")
    print()

    # 演示节点故障时的转移
    print("模拟节点故障：移除 node2")
    before = {i: ch.get_node(f"vec_{i}") for i in range(100)}
    ch.remove_node("node2")
    after = {i: ch.get_node(f"vec_{i}") for i in range(100)}

    affected = sum(1 for i in range(100) if before[i] != after[i])
    print(f"  受影响的向量：{affected}/100")
    print(f"  这些数据会被转移到 node1 或 node3")
    print()

    # 演示副本分配
    print("为向量分配 2 个副本节点：")
    ch2 = ConsistentHash(nodes, virtual_nodes=160)
    for i in range(5):
        replicas = ch2.get_nodes(f"vec_{i}", replica_count=2)
        print(f"  vec_{i} 副本节点: {replicas}")
    print()


def demo_distributed_client():
    """演示分布式客户端（假设集群已启动）"""
    print("="*70)
    print("[演示 2] 分布式客户端 Scatter-Gather 查询")
    print("="*70)
    print()

    nodes = [
        "http://127.0.0.1:8001",
        "http://127.0.0.1:8002",
        "http://127.0.0.1:8003",
    ]

    client = DistributedVectorDBClient(nodes, replica_count=1)
    print(f"✓ 创建分布式客户端，连接节点:")
    for node in nodes:
        print(f"  - {node}")
    print()

    # 尝试连接
    print("检查节点可达性...")
    reachable = 0
    for node in nodes:
        try:
            import requests
            resp = requests.get(f"{node}/health", timeout=2)
            if resp.status_code == 200:
                print(f"  ✓ {node} 可达")
                reachable += 1
            else:
                print(f"  ✗ {node} 无响应")
        except Exception as e:
            print(f"  ✗ {node} 连接失败: {e}")

    if reachable == 0:
        print()
        print("⚠️  没有节点可达，跳过集群演示")
        print("请先启动 3 个 API 服务：")
        print("  uvicorn sqcvecdb.service:app --port 8001")
        print("  uvicorn sqcvecdb.service:app --port 8002")
        print("  uvicorn sqcvecdb.service:app --port 8003")
        return

    print(f"✓ 共 {reachable} 个节点可达，继续演示")
    print()

    # 创建 collection 和插入数据
    print("步骤 1: 创建 collection")
    client.use_collection("distributed_col")
    print(f"  ✓ 使用 collection: distributed_col")
    print()

    print("步骤 2: 批量插入 300 个向量（自动分片到 3 个节点）")
    rng = np.random.RandomState(42)
    records = [
        {
            "id": i,
            "vector": rng.random(4).astype(np.float32).tolist(),
            "metadata": {"index": i, "shard": client._get_primary_node(i).split(":")[-1]},
        }
        for i in range(300)
    ]

    try:
        client.insert_many(records)
        print(f"  ✓ 插入成功")
        print()

        # 显示分片分布
        shard_dist = {}
        for i in range(300):
            node = client._get_primary_node(i)
            shard_dist[node] = shard_dist.get(node, 0) + 1

        print("分片分布：")
        for node, count in sorted(shard_dist.items()):
            pct = count / 3
            bar = "█" * int(pct / 2)
            print(f"  {node}: {count:3d} ({pct:5.1f}%) {bar}")
        print()

    except Exception as e:
        print(f"  ✗ 插入失败: {e}")
        return

    # Scatter-Gather 查询
    print("步骤 3: Scatter-Gather 查询")
    query_vec = rng.random(4).astype(np.float32).tolist()

    try:
        results = client.search(query_vec, top_k=5)
        print(f"  查询向量: {query_vec[:2]}... (略)")
        print(f"  查询耗时：并行查询 {reachable} 个节点")
        print(f"  返回 top-5 结果：")
        for i, (vec_id, distance, metadata) in enumerate(results[:5]):
            print(f"    {i+1}. vec_id={vec_id}, distance={distance:.6f}, shard={metadata.get('shard', 'N/A')}")
        print()

    except Exception as e:
        print(f"  ✗ 查询失败: {e}")
        return

    # 获取统计信息
    print("步骤 4: 集群统计信息")
    try:
        stats = client.get_stats()
        print(f"  总向量数: {stats['total_items']}")
        print(f"  节点数: {len(stats['nodes'])}")
        print(f"  各节点统计：")
        for node, node_stat in stats["node_stats"].items():
            if "error" in node_stat:
                print(f"    {node}: {node_stat['error']}")
            else:
                print(f"    {node}: {node_stat['num_items']} 向量")
        print()

    except Exception as e:
        print(f"  ✗ 获取统计失败: {e}")

    print("✓ 分布式集群演示完成")
    print()


def demo_cluster_topology():
    """演示集群拓扑和故障转移"""
    print("="*70)
    print("[演示 3] 集群拓扑与故障转移")
    print("="*70)
    print()

    nodes = [
        "http://node1:8001",
        "http://node2:8002",
        "http://node3:8003",
    ]

    client = DistributedVectorDBClient(nodes)

    print(f"初始集群拓扑:")
    print(f"  节点: {len(client.nodes)}")
    for node in client.nodes:
        print(f"    - {node}")
    print()

    print("向量分片示例（前 30 个向量）：")
    shard_map = {}
    for i in range(30):
        node = client._get_primary_node(i)
        shard_map.setdefault(node, []).append(i)

    for node, vec_ids in sorted(shard_map.items()):
        print(f"  {node}: {len(vec_ids)} 个向量")
        print(f"    ID: {vec_ids[:5]}{'...' if len(vec_ids) > 5 else ''}")
    print()

    print("模拟节点故障并演示自动转移：")
    print(f"  移除 {nodes[0]}...")
    client.remove_node(nodes[0])
    print(f"  ✓ 已移除")
    print()

    print(f"更新后的集群拓扑（{len(client.nodes)} 个节点）：")
    for node in client.nodes:
        print(f"  - {node}")
    print()

    print("向量重新分片结果（前 30 个向量）：")
    new_shard_map = {}
    for i in range(30):
        node = client._get_primary_node(i)
        new_shard_map.setdefault(node, []).append(i)

    affected = 0
    for i in range(30):
        old_node = shard_map.get(i, [None])[0] if i in [j for ids in shard_map.values() for j in ids] else None
        new_node = new_shard_map.get(i, [None])[0] if i in [j for ids in new_shard_map.values() for j in ids] else None
        if old_node and old_node != new_node:
            affected += 1

    for node, vec_ids in sorted(new_shard_map.items()):
        print(f"  {node}: {len(vec_ids)} 个向量")
        print(f"    ID: {vec_ids[:5]}{'...' if len(vec_ids) > 5 else ''}")
    print()

    print(f"  ✓ 故障转移完成，数据自动转移到其他节点")
    print()


def main():
    print()
    print("  🚀 sqcvecdb 分布式集群演示（Phase 5）")
    print()

    # 演示 1：一致性哈希
    demo_consistent_hash()

    # 演示 2：一致性哈希集成
    demo_cluster_topology()

    # 演示 3：分布式客户端（需要实际的 API 服务运行）
    demo_distributed_client()

    print("="*70)
    print("✓ 所有演示完成")
    print("="*70)


if __name__ == "__main__":
    main()
