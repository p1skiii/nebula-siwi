import os
from flask import Flask, jsonify, request
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from siwi.bot import bot # 假设 bot.py 在 siwi 目录下
from siwi.feature_store import get_entity_embedding # 假设 feature_store.py 在 siwi 目录下
from siwi.subgraph_sampler import SubgraphSampler # 假设 subgraph_sampler.py 在 siwi 目录下

# --- Flask App Initialization ---
app = Flask(__name__)

# --- NebulaGraph Connection Configuration ---
def parse_nebula_graphd_endpoint():
    ng_endpoints_str = os.environ.get('NG_ENDPOINTS', '127.0.0.1:9669').split(",")
    ng_endpoints = []
    for endpoint in ng_endpoints_str:
        if endpoint:
            parts = endpoint.split(":")
            if len(parts) == 2:
                ng_endpoints.append((parts[0], int(parts[1])))
    if not ng_endpoints: # 提供一个默认值，如果环境变量解析失败
        ng_endpoints.append(('127.0.0.1', 9669))
    return ng_endpoints

ng_config = Config()
ng_config.max_connection_pool_size = int(os.environ.get('NG_MAX_CONN_POOL_SIZE', 10))
ng_endpoints = parse_nebula_graphd_endpoint()
connection_pool = ConnectionPool()
# 初始化连接池，确保在应用启动前完成
if not connection_pool.init(ng_endpoints, ng_config):
    raise RuntimeError("Failed to initialize NebulaGraph connection pool")

# --- Global Variables ---
siwi_bot = bot.SiwiBot(connection_pool) # 初始化 SiwiBot

# --- Route Definitions ---
@app.route("/")
def root():
    return "Hey There?"

@app.route("/query", methods=["POST"])
def query_route(): # 避免与内置的 query 重名
    request_data = request.get_json()
    question = request_data.get("question", "")
    if question:
        answer = siwi_bot.query(question)
    else:
        answer = "Sorry, what did you say?"
    return jsonify({"answer": answer})

@app.route("/api/v1/entity/<entity_tag>/<entity_id>/embedding", methods=["GET"])
def get_entity_embedding_api(entity_tag, entity_id):
    try:
        embedding_value = get_entity_embedding(entity_id, entity_tag)
        if embedding_value is None:
            return jsonify({
                "success": False,
                "error": f"无法找到实体 {entity_tag}:{entity_id} 的embedding1值"
            }), 404
        return jsonify({
            "success": True,
            "entity_id": entity_id,
            "entity_type": entity_tag,
            "embedding": embedding_value
        })
    except Exception as e:
        import traceback
        print(f"Error in get_entity_embedding_api: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/v1/subgraph/<entity_id>/<int:n_hops>", methods=["GET"])
def get_subgraph(entity_id, n_hops):
    try:
        n_hops = min(n_hops, 3)
        space_name = request.args.get("space", "basketballplayer")
        max_nodes = int(request.args.get("max_nodes", 1000))
        
        sampler = SubgraphSampler(connection_pool) # 传递连接池
        subgraph_data = sampler.sample_subgraph(
            center_vid=entity_id,
            n_hops=n_hops,
            space_name=space_name,
            max_nodes=max_nodes
        )
        
        # 确保 subgraph_data 中的必要字段存在且格式正确
        center_idx = subgraph_data.get('center_node_idx', 0)
        num_nodes = subgraph_data.get('num_nodes', 0)
        edge_index = subgraph_data.get('edge_index')
        num_edges = edge_index.shape[1] if edge_index is not None and hasattr(edge_index, 'shape') else 0
        
        nodes_list = []
        if 'idx_to_vid' in subgraph_data and isinstance(subgraph_data['idx_to_vid'], list):
            for idx, vid_val in enumerate(subgraph_data['idx_to_vid']):
                node_info = {
                    "idx": idx,
                    "vid": vid_val,
                    "type": subgraph_data.get('node_types', {}).get(vid_val, "unknown"),
                    "name": subgraph_data.get('node_features', {}).get(vid_val, {}).get('name', "")
                }
                nodes_list.append(node_info)

        edges_list = []
        if edge_index is not None and num_edges > 0 and 'idx_to_vid' in subgraph_data:
            for i in range(num_edges):
                src_idx = int(edge_index[0, i])
                tgt_idx = int(edge_index[1, i])
                edge_info = {
                    "source_idx": src_idx,
                    "target_idx": tgt_idx,
                    "source_vid": subgraph_data['idx_to_vid'][src_idx] if src_idx < len(subgraph_data['idx_to_vid']) else "unknown_src",
                    "target_vid": subgraph_data['idx_to_vid'][tgt_idx] if tgt_idx < len(subgraph_data['idx_to_vid']) else "unknown_tgt"
                }
                edges_list.append(edge_info)

        result = {
            "success": True,
            "subgraph": {
                "center_node": entity_id,
                "center_idx": int(center_idx),
                "num_nodes": int(num_nodes),
                "num_edges": int(num_edges),
                "nodes": nodes_list,
                "edges": edges_list
            }
        }
        return jsonify(result)
    except Exception as e:
        import traceback
        print(f"Error in get_subgraph: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/v1/pyg/<entity_id>/<int:n_hops>", methods=["GET"])
def get_pyg_subgraph(entity_id, n_hops):
    try:
        n_hops = min(n_hops, 3)
        space_name = request.args.get("space", "basketballplayer")
        
        sampler = SubgraphSampler(connection_pool) # 传递连接池
        subgraph_data = sampler.sample_subgraph(
            center_vid=entity_id,
            n_hops=n_hops,
            space_name=space_name
        )

        edge_index = subgraph_data.get('edge_index')
        num_edges = edge_index.shape[1] if edge_index is not None and hasattr(edge_index, 'shape') else 0
        
        edge_index_list = []
        if edge_index is not None and num_edges > 0:
             edge_index_list = [
                [int(edge_index[0, i]), int(edge_index[1, i])]
                for i in range(num_edges)
            ]
        
        node_features_list = [0.0] * subgraph_data.get('num_nodes', 0)
        if 'idx_to_vid' in subgraph_data and 'node_features' in subgraph_data:
            for idx, vid_val in enumerate(subgraph_data['idx_to_vid']):
                node_feature_data = subgraph_data['node_features'].get(vid_val, {})
                if 'embedding' in node_feature_data and hasattr(node_feature_data['embedding'], 'item'):
                    node_features_list[idx] = float(node_feature_data['embedding'].item())
        
        return jsonify({
            "success": True,
            "pyg_data": {
                "x": node_features_list,
                "edge_index": edge_index_list,
                "num_nodes": subgraph_data.get('num_nodes', 0),
                "center_node_idx": int(subgraph_data.get('center_node_idx', 0)),
                "idx_to_vid": subgraph_data.get('idx_to_vid', [])
            }
        })
    except Exception as e:
        import traceback
        print(f"Error in get_pyg_subgraph: {traceback.format_exc()}")
        return jsonify({"success": False, "error": str(e)}), 500

def run_app():
    # 打印所有已注册的路由，用于调试
    print("\n=== 已注册的路由 ===")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint}: {rule.rule} Methods: {list(rule.methods)}")
    print("==================\n")
    
    app.run(host="0.0.0.0", port=5000, debug=True)

@app.route("/debug/routes")
def debug_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "path": str(rule)
        })
    return jsonify({
        "routes": routes,
        "total": len(routes)
    })

# 然后在应用启动时打印路由
print("\n=== 已注册的路由 ===")
for rule in app.url_map.iter_rules():
    print(f"{rule.endpoint}: {rule.rule}")
print("==================\n")

if __name__ == "__main__":
    try:
        run_app()
    finally:
        if connection_pool:
            connection_pool.close()