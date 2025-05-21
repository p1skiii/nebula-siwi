# filepath: /Users/wang/i/nebula-siwi/test_api.py
import sys
import os

# 确保src目录在Python路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from siwi.app import app

# 使用测试客户端
client = app.test_client()

# 测试根路径
response = client.get('/')
print(f"根路径响应: {response.data.decode()}")

# 测试embedding API
response = client.get('/api/v1/entity/player/player100/embedding')
print(f"embedding API 状态码: {response.status_code}")
print(f"embedding API 响应: {response.data.decode()}")