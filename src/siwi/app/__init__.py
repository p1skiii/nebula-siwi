import os

from flask import Flask, jsonify, request
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from siwi.bot import bot
from siwi.feature_store import get_entity_embedding, convert_embedding_to_tensor


app = Flask(__name__)
# 初始化全局变量
siwi_bot = None


@app.route("/")
def root():
    return "Hey There?"


@app.route("/query", methods=["POST"])
def query():
    request_data = request.get_json()
    question = request_data.get("question", "")
    if question:
        answer = siwi_bot.query(request_data.get("question", ""))
    else:
        answer = "Sorry, what did you say?"
    return jsonify({"answer": answer})


@app.route("/api/v1/entity/<entity_tag>/<entity_id>/features", methods=["GET"])
def get_entity_features(entity_tag, entity_id):
    """获取指定实体的特征向量
    
    Args:
        entity_tag: 实体的标签，例如 'player'
        entity_id: 实体的ID
        
    Returns:
        JSON格式的实体特征向量
    """
    try:
        # 获取embedding列表
        embedding_list = get_entity_embedding(entity_id, entity_tag)
        
        if embedding_list is None:
            return jsonify({
                "success": False,
                "error": f"无法找到实体 {entity_tag}:{entity_id} 的embedding"
            }), 404
        
        # 转换为Tensor (可选，这里我们只是检查转换是否成功)
        tensor = convert_embedding_to_tensor(embedding_list)
        if tensor is None:
            return jsonify({
                "success": False,
                "error": "embedding转换为tensor失败"
            }), 500
        
        # 返回embedding列表，前端可以根据需要转换为tensor
        return jsonify({
            "success": True,
            "entity_id": entity_id,
            "entity_type": entity_tag,
            "features": embedding_list,
            "dimension": len(embedding_list)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def parse_nebula_graphd_endpoint():
    ng_endpoints_str = os.environ.get(
        'NG_ENDPOINTS', '127.0.0.1:9669,').split(",")
    ng_endpoints = []
    for endpoint in ng_endpoints_str:
        if endpoint:
            parts = endpoint.split(":")  # we dont consider IPv6 now
            ng_endpoints.append((parts[0], int(parts[1])))
    return ng_endpoints


ng_config = Config()
ng_config.max_connection_pool_size = int(
    os.environ.get('NG_MAX_CONN_POOL_SIZE', 10))
ng_endpoints = parse_nebula_graphd_endpoint()
connection_pool = ConnectionPool()

if __name__ == "__main__":
    connection_pool.init(ng_endpoints, ng_config)
    siwi_bot = bot.SiwiBot(connection_pool)
    try:
        app.run(host="0.0.0.0", port=5000)
    finally:
        connection_pool.close()
else:
    connection_pool.init(ng_endpoints, ng_config)
    siwi_bot = bot.SiwiBot(connection_pool)
