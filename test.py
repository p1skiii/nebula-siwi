#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time
import numpy as np
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config

# NebulaGraph 连接配置
NEBULA_HOSTS = [('127.0.0.1', 9669)]
NEBULA_USER = 'root'
NEBULA_PASSWORD = 'nebula'
NEBULA_SPACE = 'basketballplayer'

# 初始化连接
connection_pool = ConnectionPool()
config = Config()
config.max_connection_pool_size = 10
if not connection_pool.init(NEBULA_HOSTS, config):
    print("连接池初始化失败")
    sys.exit(1)

with connection_pool.session_context(NEBULA_USER, NEBULA_PASSWORD) as session:
    # 使用图空间
    resp = session.execute(f'USE {NEBULA_SPACE}')
    if not resp.is_succeeded():
        print("切换图空间失败:", resp.error_msg())
        sys.exit(1)
    
    # 步骤1: 检查TAG结构，查看实际的Tag名称
    desc_query = 'SHOW TAGS'
    resp = session.execute(desc_query)
    if resp.is_succeeded():
        print("图空间中的所有标签:")
        print(resp)
    else:
        print("获取标签列表失败:", resp.error_msg())
    
    # 查看具体的player标签结构
    desc_player = 'DESCRIBE TAG player'
    resp = session.execute(desc_player)
    if resp.is_succeeded():
        print("\nplayer标签的结构:")
        print(resp)
    else:
        print("获取player标签结构失败:", resp.error_msg())
    
    # 步骤2: 为player标签添加embedding属性
    alter_query = 'ALTER TAG player ADD (embedding double)'
    resp = session.execute(alter_query)
    if resp.is_succeeded():
        print("\n成功添加embedding属性")
    else:
        print("添加属性失败 (可能已存在):", resp.error_msg())
    
    # 步骤3: 等待Schema变更生效 (这是关键修改)
    print("\n等待Schema变更生效...")
    time.sleep(5)  # 等待5秒钟让Schema变更生效
    
    # 再次检查标签结构，确认embedding属性已添加
    resp = session.execute('DESCRIBE TAG player')
    if resp.is_succeeded():
        print("\n更新后的player标签结构:")
        print(resp)
    else:
        print("获取player标签结构失败:", resp.error_msg())
    
    # 步骤4: 获取所有球员的ID
    player_query = 'MATCH (v:player) RETURN id(v) AS id'
    resp = session.execute(player_query)
    if not resp.is_succeeded():
        print("获取球员列表失败:", resp.error_msg())
        sys.exit(1)
    
    # 提取球员ID列表
    player_ids = []
    if resp.row_size() > 0:
        for i in range(resp.row_size()):
            player_id = resp.row_values(i)[0].as_string()
            player_ids.append(player_id)
        print(f"\n找到 {len(player_ids)} 个球员")
    else:
        print("没有找到球员")
        sys.exit(1)
    
    # 步骤5: 尝试不同的更新方式 - 使用UPSERT而不是UPDATE
    updated_count = 0
    for player_id in player_ids:
        random_value = round(np.random.uniform(30, 40), 2)
        
        # 使用UPSERT而不是UPDATE
        upsert_query = f'UPSERT VERTEX ON player "{player_id}" SET embedding = {random_value}'
        resp = session.execute(upsert_query)
        
        if resp.is_succeeded():
            updated_count += 1
        else:
            print(f"更新球员 {player_id} 失败: {resp.error_msg()}")
    
    print(f"\n成功更新 {updated_count}/{len(player_ids)} 个球员的embedding属性")
    
    # 步骤6: 验证几个球员的embedding值
    if player_ids and updated_count > 0:
        sample_ids = player_ids[:3]  # 取前3个球员作为样本
        sample_ids_str = '", "'.join(sample_ids)
        verify_query = f'FETCH PROP ON player "{sample_ids_str}" YIELD properties(vertex).name, properties(vertex).embedding'
        resp = session.execute(verify_query)
        
        if resp.is_succeeded():
            print("\n验证几个球员的embedding值:")
            print(resp)
        else:
            print("验证失败:", resp.error_msg())

# 关闭连接池
connection_pool.close()