"""
测试PyG与NebulaGraph集成功能
"""

import os
import sys
# 确保src目录在Python路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from siwi.pyg_integration import NebulaToTorch

def test_node_features():
    """测试获取节点特征"""
    print("\n=== 测试获取节点特征 ===")
    
    # 创建集成类实例
    converter = NebulaToTorch()
    
    # 测试获取球员特征
    player_ids = ["player142", "player117"]  # 姚明和库里
    features = converter.get_node_features(player_ids, "player")
    
    print(f"获取到{len(player_ids)}个球员的特征:")
    for i, player_id in enumerate(player_ids):
        print(f"- {player_id}: {features[i].item() if i < len(features) else 'N/A'}")
    
    return features

def test_subgraph():
    """测试获取子图"""
    print("\n=== 测试获取子图 ===")
    
    # 创建集成类实例
    converter = NebulaToTorch()
    
    # 测试获取以姚明为中心的子图
    center_nodes = ["player142"]  # 姚明
    subgraph = converter.get_subgraph(center_nodes, n_hops=1)
    
    print(f"获取到以{center_nodes[0]}为中心的子图:")
    print(f"- 节点数: {subgraph['num_nodes']}")
    print(f"- 边数: {len(subgraph['edge_index'][0]) if subgraph['edge_index'] else 0}")
    print(f"- 节点ID: {subgraph['node_ids']}")
    
    return subgraph

if __name__ == "__main__":
    print("=== 开始测试PyG集成功能 ===")
    
    # 测试获取节点特征
    features = test_node_features()
    
    # 测试获取子图
    subgraph = test_subgraph()
    
    print("\n=== 测试完成 ===")