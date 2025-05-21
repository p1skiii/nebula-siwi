import os
import sys

# 添加项目根目录到系统路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))
sys.path.insert(0, os.path.join(BASE_DIR, 'src', 'siwi'))

# 导入应用
from src.siwi.app import app, init_app

# 初始化应用（连接池和bot）
init_app()

# app是Flask对象
application = app

# 启动命令: gunicorn --bind :5000 wsgi:application --workers 1 --threads 1 --timeout 60
