import sys
import os
import requests
import json

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

def test_subgraph_api():
    """测试子图采样API端点"""
    # 基础URL
    base_url = "http://localhost:5000"
    
    # 测试接口
    endpoints = [
        "/api/v1/subgraph/player142/1",  # 姚明1跳子图
        "/api/v1/subgraph/player100/1",  # 另一个球员1跳子图
        "/api/v1/subgraph/player142/2?max_nodes=50"  # 姚明2跳子图,限制节点数
    ]
    
    for endpoint in endpoints:
        print(f"\n测试端点: {endpoint}")
        
        # 发送GET请求
        response = requests.get(f"{base_url}{endpoint}")
        
        # 打印状态码
        print(f"状态码: {response.status_code}")
        
        # 如果请求成功，打印部分结果
        if response.status_code == 200:
            data = response.json()
            print(f"成功: {data.get('success', False)}")
            
            if 'subgraph' in data:
                subgraph = data['subgraph']
                print(f"节点数: {subgraph.get('num_nodes', 0)}")
                print(f"边数: {subgraph.get('num_edges', 0)}")
                
                # 打印前5个节点信息
                nodes = subgraph.get('nodes', [])
                print(f"\n前{min(5, len(nodes))}个节点:")
                for node in nodes[:5]:
                    print(f"- {node.get('vid', '')}: {node.get('name', '')}")
        else:
            print(f"错误: {response.text}")
    
    print("\n测试完成!")

def test_api(host="localhost", port=5000, path="/"):
    """测试访问API端点"""
    print(f"正在连接到 {host}:{port}{path}")
    
    conn = http.client.HTTPConnection(host, port)
    conn.request("GET", path)
    response = conn.getresponse()
    
    print(f"状态: {response.status} {response.reason}")
    data = response.read()
    print(f"响应内容: {data.decode('utf-8')[:100]}...")
    conn.close()


def test_subgraph_api():
    """测试子图采样API端点"""
    print("\n=== 开始测试子图API端点 ===")
    
    # 基础URL，使用完整URL
    base_url = "http://localhost:5000"  # 使用localhost
    
    # 先测试根路径，确认服务器正在运行
    print("\n测试根路径:")
    try:
        root_response = requests.get(f"{base_url}/")
        print(f"根路径状态码: {root_response.status_code}")
        print(f"根路径响应: {root_response.text[:50]}...")
        
        if root_response.status_code != 200:
            print("警告: 服务器运行异常，根路径返回非200状态码")
            return
    except Exception as e:
        print(f"无法连接到服务器: {e}")
        print("请确保Flask服务器正在运行")
        return
    
    # 测试子图API端点
    endpoints = [
        "/api/v1/subgraph/player142/1"
    ]
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        print(f"\n测试端点: {url}")
        
        # 发送GET请求，添加详细Headers以便于调试
        try:
            headers = {
                'User-Agent': 'SubgraphAPITest/1.0',
                'Accept': 'application/json'
            }
            response = requests.get(url, headers=headers, timeout=30)
            print(f"状态码: {response.status_code}")
            
            # 打印响应头信息
            print("响应头:")
            for header, value in response.headers.items():
                print(f"  {header}: {value}")
            
            # 打印响应内容
            print(f"响应内容: {response.text[:200]}...")
            
            # 解析JSON（如果成功）
            if response.status_code == 200:
                try:
                    data = response.json()
                    print(f"成功解析JSON响应")
                except json.JSONDecodeError:
                    print(f"警告: 响应不是有效的JSON")
        except Exception as e:
            print(f"请求异常: {e}")
    
    print("=== 子图API测试完成 ===")



if __name__ == "__main__":
    print("测试子图采样模块...")
    
    # 测试子图采样
    test_sample_subgraph()
    
    # 测试转换为PyG
    test_convert_to_pyg()
    
    test_subgraph_api()

    test_subgraph_api()

    print("\n所有测试完成!")

