# Nebula-SIWI: NebulaGraph 与 PyTorch Geometric 集成

Nebula-SIWI 项目实现了 PyTorch Geometric (PyG) 的远程后端接口，使 PyG 能够直接与 NebulaGraph 图数据库交互，为图机器学习提供高效的数据访问层。

## 项目概述

该项目提供了一个桥接层，实现了 PyG 的 `FeatureStore` 和 `GraphStore` 接口，使得 PyG 可以直接从 NebulaGraph 数据库中读取和操作图数据，而无需将整个图导出和加载到内存中。这对于处理大规模图数据特别有用。

### 核心功能

- **NebulaFeatureStore**: 实现了 PyG 的 `FeatureStore` 接口，负责从 NebulaGraph 获取节点特征
- **NebulaGraphStore**: 实现了 PyG 的 `GraphStore` 接口，负责从 NebulaGraph 获取图结构（边）
- **SubgraphSampler**: 高效地从 NebulaGraph 采样子图，支持多跳邻居采样
- **SimpleNeighborLoader**: 提供简化的数据加载器，支持 PyG 的邻居采样模式

## 架构设计

```
┌─────────────────────────────┐
│       PyTorch Geometric     │
│    (图神经网络模型和工具)      │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│ 远程后端接口 (Remote Backend) │
│ FeatureStore    GraphStore  │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│     Nebula-SIWI 桥接层       │
│                             │
│  ├── NebulaFeatureStore     │
│  ├── NebulaGraphStore       │
│  ├── SubgraphSampler        │
│  └── SimpleNeighborLoader   │
└──────────────┬──────────────┘
               │
┌──────────────▼──────────────┐
│        NebulaGraph          │
│      (分布式图数据库)         │
└─────────────────────────────┘
```

## 技术细节

### NebulaFeatureStore

实现了 PyG 的 `FeatureStore` 抽象类，提供以下核心功能：

- **_get_tensor**: 从 NebulaGraph 获取节点特征
- **_get_tensor_size**: 获取特征张量的大小
- **_put_tensor**: 将特征存储到 NebulaGraph
- **_remove_tensor**: 从 NebulaGraph 移除特征
- **get_all_tensor_attrs**: 获取所有可用特征属性

### NebulaGraphStore

实现了 PyG 的 `GraphStore` 抽象类，提供以下核心功能：

- **_get_edge_index**: 获取边索引，返回 COO 格式的边表示
- **_put_edge_index**: 将边索引存储到 NebulaGraph
- **_remove_edge_index**: 从 NebulaGraph 移除边索引
- **get_all_edge_attrs**: 获取所有可用的边类型

### SubgraphSampler

负责从 NebulaGraph 高效地采样子图：

- 支持 `n_hops` 参数指定采样的跳数
- 支持通过 nGQL 查询语言高效获取子图结构
- 自动处理 NebulaGraph 的字符串 VID 和 PyG 的数字索引之间的映射
- 提供边类型过滤和节点类型识别

### ID 映射机制

项目实现了灵活的 ID 映射机制，解决了 NebulaGraph 使用字符串 VID 而 PyG 使用数字索引的不兼容问题：

- `id_to_idx` 和 `idx_to_id` 提供双向映射
- 支持自定义 ID 映射器函数
- 确保在所有操作中保持 ID 映射的一致性

## 使用示例

### 基本使用

```python
from siwi.pyg_integration import NebulaToTorch

# 创建集成类实例
converter = NebulaToTorch(space_name="basketballplayer")

# 获取节点特征
player_ids = ["player142", "player117"]  # 姚明和库里
features = converter.get_node_features(player_ids, "player")

# 获取子图
center_nodes = ["player142"]  # 以姚明为中心
subgraph = converter.get_subgraph(center_nodes, n_hops=1)

# 子图信息
print(f"节点数: {subgraph['num_nodes']}")
print(f"边数: {len(subgraph['edge_index'][0])}")
```

### 与 PyG 模型集成

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from siwi.pyg_integration import NebulaToTorch

# 定义GCN模型
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 获取数据
converter = NebulaToTorch()
subgraph = converter.get_subgraph(["player142"], n_hops=2)

# 转换为PyG数据格式
x = torch.tensor(subgraph["features"], dtype=torch.float)
edge_index = torch.tensor(subgraph["edge_index"], dtype=torch.long)

# 初始化并运行模型
model = GCN(in_channels=x.size(1), hidden_channels=16, out_channels=2)
out = model(x, edge_index)
```

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖:
- nebula3-python: NebulaGraph Python客户端
- torch: PyTorch深度学习框架
- torch-geometric: PyTorch几何图神经网络库

## 测试

可以使用提供的测试脚本验证集成功能是否正常工作：

```bash
python test_pyg_integration.py
```

## 注意事项

- 确保 NebulaGraph 服务正在运行，并配置了正确的连接参数
- 对于大规模图，建议适当调整跳数和最大节点数限制
- 本项目主要用于演示和研究目的，生产环境中可能需要进一步优化
