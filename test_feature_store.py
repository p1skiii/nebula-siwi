import sys
import os

# 确保src目录在Python路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# 导入feature_store模块
from siwi.feature_store import (
    get_entity_embedding,
    convert_embedding_to_tensor,
    get_entity_embedding_tensor
)

def test_get_embedding():
    print("\n===== 测试embedding1获取 =====")
    player_id = "player100"
    print(f"获取球员 {player_id} 的embedding1...")
    embedding = get_entity_embedding(player_id)
    print(f"embedding1类型: {type(embedding)}")
    print(f"embedding1值: {embedding}")
    return embedding

def test_tensor_conversion():
    print("\n===== 测试tensor转换 =====")
    # 获取一个embedding值
    player_id = "player100"
    embedding = get_entity_embedding(player_id)
    
    if embedding is not None:
        print(f"原始embedding1值: {embedding}")
        tensor = convert_embedding_to_tensor(embedding)
        print(f"转换后的tensor: {tensor}")
        print(f"tensor形状: {tensor.shape if tensor is not None else '无'}")
    else:
        print(f"无法获取球员 {player_id} 的embedding1")

def test_direct_tensor():
    print("\n===== 测试直接获取tensor =====")
    player_id = "player100"
    print(f"直接获取球员 {player_id} 的embedding1 tensor...")
    tensor = get_entity_embedding_tensor(player_id)
    
    if tensor is not None:
        print(f"获取的tensor: {tensor}")
        print(f"tensor形状: {tensor.shape}")
    else:
        print(f"无法获取球员 {player_id} 的embedding1 tensor")

if __name__ == "__main__":
    print("开始测试feature_store模块...")
    
    # 测试embedding1获取
    embedding = test_get_embedding()
    
    # 测试tensor转换
    test_tensor_conversion()
    
    # 测试直接获取tensor
    test_direct_tensor()
    
    print("\n测试完成!")