import torch
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

from torch_geometric.data import FeatureStore
from torch_geometric.graphstore import GraphStore

from siwi.feature_store import get_entity_embedding, get_nebula_connection_pool
from siwi.subgraph_sampler import SubgraphSampler

class NebulaFeatureStore(FeatureStore):
    def __init__(self, space_name: str = "basketballplayer"):
        """初始化NebulaFeatureStore
        
        Args:
            space_name: NebulaGraph图空间名称
        """
        self.space_name = space_name
        self.connection_pool = get_nebula_connection_pool()

    def get_tensor(self, group: str, name: str, index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """获取指定节点的特征
        
        Args:
            group: 节点类型，例如"player"、"team"
            name: 特征名称，例如"embedding1"
            index: 节点索引张量
            
        Returns:
            特征张量
        """
        print(f"获取{group}节点的{name}特征，索引大小: {index.size() if index is not None else 'None'}")
        
        # 如果未提供索引，返回空张量
        if index is None:
            return torch.tensor([], dtype=torch.float)
        
        # 将索引转换为字符串列表
        # 注意：在实际应用中，您可能需要更复杂的ID映射
        node_ids = [str(idx.item()) for idx in index]
        
        # 获取每个节点的特征
        features = []
        for node_id in node_ids:
            # 调用功能1获取特征
            embedding = get_entity_embedding(node_id, group, name)
            
            # 如果找不到特征，使用零向量
            if embedding is None:
                features.append(torch.tensor([0.0], dtype=torch.float))
            else:
                # 将特征转换为张量
                features.append(torch.tensor([float(embedding)], dtype=torch.float))
        
        # 堆叠所有特征
        if features:
            return torch.cat(features, dim=0)
        else:
            return torch.zeros((len(node_ids), 1), dtype=torch.float)