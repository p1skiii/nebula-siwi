import torch
from typing import List, Dict, Any, Optional, Tuple, Union

from torch_geometric.data import Data

from siwi.remote_backend import NebulaFeatureStore, NebulaGraphStore

class SimpleNeighborLoader:
    """简化版的邻居加载器
    
    这个类演示如何使用NebulaFeatureStore和NebulaGraphStore加载数据
    """
    
    def __init__(self, feature_store: NebulaFeatureStore, 
                graph_store: NebulaGraphStore,
                node_type: str = "player",
                edge_type: str = "follow"):
        """初始化加载器
        
        Args:
            feature_store: 特征存储
            graph_store: 图存储
            node_type: 节点类型
            edge_type: 边类型
        """
        self.feature_store = feature_store
        self.graph_store = graph_store
        self.node_type = node_type
        self.edge_type = edge_type
    
    def load_data(self, seed_nodes: List[str], node_indices: List[int], num_hops: int = 1) -> Data:
        """加载以种子节点为中心的子图数据
        
        Args:
            seed_nodes: 种子节点ID列表
            node_indices: 节点索引列表（对应于seed_nodes）
            num_hops: 跳数
            
        Returns:
            PyG Data对象
        """
        print(f"为{len(seed_nodes)}个种子节点加载{num_hops}跳邻居")
        
        # 1. 创建节点索引张量
        node_indices_tensor = torch.tensor(node_indices, dtype=torch.long)
        
        # 2. 获取节点特征
        node_features = self.feature_store.get_tensor(
            group=self.node_type,
            name="embedding1",
            index=node_indices_tensor
        )
        
        # 3. 获取边结构（这是简化版，实际应用需要更复杂的逻辑）
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        # 4. 创建PyG Data对象
        data = Data(
            x=node_features,
            edge_index=edge_index,
            num_nodes=len(seed_nodes)
        )
        
        # 添加节点ID映射以便于查询
        data.node_ids = seed_nodes
        
        return data