import sys
import os

# 确保src目录在Python模块搜索路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 直接从src目录导入
from  siwi.subgraph_sampler import SubgraphSampler

def test_sample_subgraph():
    """测试子图采样功能"""
    # 姚明的ID是 "player142"
    center_vid = "player142"
    n_hops = 2  # 2跳子图
    
    # 创建采样器
    sampler = SubgraphSampler()
    
    # 采样子图
    subgraph = sampler.sample_subgraph(
        center_vid=center_vid,
        n_hops=n_hops,
        space_name="basketballplayer"
    )
    
    # 打印子图信息
    print(f"子图信息:")
    print(f"- 中心节点: {center_vid} (索引: {subgraph['center_node_idx']})")
    print(f"- 节点数量: {subgraph['num_nodes']}")
    print(f"- 边数量: {subgraph['edge_index'].shape[1]}")
    
    # 打印边类型信息
    print("\n边类型信息:")
    for edge_type, edge_index in subgraph['edge_indices_by_type'].items():
        print(f"- {edge_type}: {edge_index.shape[1]}条边")
    
    # 获取中心节点的邻居
    center_idx = subgraph['center_node_idx']
    neighbors = []
    
    for i in range(subgraph['edge_index'].shape[1]):
        if subgraph['edge_index'][0, i] == center_idx:
            neighbor_idx = subgraph['edge_index'][1, i].item()
            neighbor_vid = subgraph['idx_to_vid'][neighbor_idx]
            neighbors.append(neighbor_vid)
    
    print(f"\n{center_vid}的直接邻居:")
    for neighbor_vid in set(neighbors):
        name = ""
        if neighbor_vid in subgraph['node_features'] and 'name' in subgraph['node_features'][neighbor_vid]:
            name = subgraph['node_features'][neighbor_vid]['name']
        print(f"- {neighbor_vid} {name}")
    
    return subgraph

def test_convert_to_pyg():
    """测试转换为PyG数据对象"""
    # 先获取子图
    subgraph = test_sample_subgraph()
    
    # 创建采样器
    sampler = SubgraphSampler()
    
    # 转换为PyG数据对象
    pyg_data = sampler.convert_to_pyg_data(subgraph)
    if pyg_data:
        print("\nPyG Data对象信息:")
        print(f"- Data.x: {pyg_data.x.shape}")
        print(f"- Data.edge_index: {pyg_data.edge_index.shape}")
        print(f"- Data.node_type: {pyg_data.node_type.shape}")
        return pyg_data
    else:
        print("\nPyG未安装，无法创建Data对象")
        return None

if __name__ == "__main__":
    print("测试子图采样模块...")
    
    # 测试子图采样
    test_sample_subgraph()
    
    # 测试转换为PyG
    test_convert_to_pyg()
    
    print("\n所有测试完成!")