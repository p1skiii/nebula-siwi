# File: /Users/wang/i/nebula-siwi/src/siwi/feature_store.py
import torch
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
# 如果你有统一的 NebulaGraph 连接管理，可以考虑复用
# 否则，这里我们定义一个独立的连接方式

# --- NebulaGraph Connection Configuration ---
# TODO: 这些配置最好从项目的统一配置文件中读取
NEBULA_HOST = '127.0.0.1'  # 你的 NebulaGraph 服务地址
NEBULA_PORT = 9669         # 你的 NebulaGraph 服务端口
NEBULA_USER = 'root'       # NebulaGraph 用户名
NEBULA_PASSWORD = 'nebula' # NebulaGraph 密码
# TODO: 确保这里的图空间名称与你的项目一致
NEBULA_GRAPH_SPACE = 'basketballplayer' # 你的图空间名称
# --- End Configuration ---

_connection_pool = None

def get_nebula_connection_pool():
    global _connection_pool
    if _connection_pool is None:
        config = Config()
        config.max_connection_pool_size = 10 # 可根据需要调整
        _connection_pool = ConnectionPool()
        # 确保 NEBULA_HOST 和 NEBULA_PORT 是正确的
        if not _connection_pool.init([(NEBULA_HOST, NEBULA_PORT)], config):
            raise RuntimeError("Failed to initialize NebulaGraph connection pool")
    return _connection_pool

def get_entity_embedding(entity_id: str, entity_tag: str = "player") -> list[float] | None:
    """
    从 NebulaGraph 中获取指定实体的 embedding 列表。
    """
    pool = get_nebula_connection_pool()
    session = None
    try:
        session = pool.get_session(NEBULA_USER, NEBULA_PASSWORD)
        session.execute(f"USE {NEBULA_GRAPH_SPACE};")
        
        # 注意：nGQL 中的 VID 通常是字符串。如果 entity_id 是整数，需要确保查询时正确引用。
        # FETCH PROP ON player "player101" YIELD properties(vertex).embedding
        query = f'FETCH PROP ON {entity_tag} "{entity_id}" YIELD properties(vertex).embedding'
        # print(f"Executing nGQL: {query}") # 用于调试
        
        result = session.execute(query)
        if not result.is_succeeded():
            print(f"Error fetching embedding for {entity_tag} '{entity_id}': {result.error_msg()}")
            return None

        if result.row_size() == 0:
            # print(f"No vertex found for {entity_tag} '{entity_id}'")
            return None
        
        # properties(vertex).embedding 结果在第一列
        embedding_value_wrapper = result.row_values(0)[0]

        if embedding_value_wrapper.is_empty() or not embedding_value_wrapper.is_list():
            # print(f"No embedding attribute found or it's not a list for {entity_tag} '{entity_id}'")
            return None
        
        embedding_list = [val.as_double() for val in embedding_value_wrapper.as_list()]
        return embedding_list

    except Exception as e:
        print(f"Exception while fetching embedding for {entity_tag} '{entity_id}': {e}")
        return None
    finally:
        if session:
            session.release()
        # 通常连接池在应用关闭时才关闭，这里暂时不关闭 session pool
        # if pool:
        #     pool.close() # 如果每次都创建新pool则需要关闭

def convert_embedding_to_tensor(embedding_list: list[float] | None) -> torch.Tensor | None:
    """
    将 embedding 列表转换为 PyTorch Tensor。
    """
    if embedding_list is None:
        return None
    try:
        return torch.tensor(embedding_list, dtype=torch.float32)
    except Exception as e:
        print(f"Error converting list to PyTorch Tensor: {e}")
        return None

# --- 模块自测试 (可选) ---
if __name__ == '__main__':
    # 确保你已经安装了 torch 和 nebula3-python
    # pip install torch nebula3-python

    # 替换为一个你已经在 NebulaGraph 中添加了 embedding 的 player ID
    test_player_id = "player101" 
    
    print(f"Attempting to fetch embedding for player: {test_player_id}")
    raw_embedding = get_entity_embedding(test_player_id, "player")

    if raw_embedding:
        print(f"  Raw embedding list: {raw_embedding}")
        tensor_feature = convert_embedding_to_tensor(raw_embedding)
        if tensor_feature is not None:
            print(f"  PyTorch Tensor: {tensor_feature}")
            print(f"  Tensor shape: {tensor_feature.shape}")
        else:
            print("  Failed to convert embedding to tensor.")
    else:
        print(f"  Could not retrieve embedding for player: {test_player_id}.")
        print(f"  Please ensure NebulaGraph is running, '{NEBULA_GRAPH_SPACE}' space exists,")
        print(f"  player '{test_player_id}' exists, and has an 'embedding' attribute of type LIST<DOUBLE>.")
