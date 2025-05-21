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
    print("成功切换到图空间:", NEBULA_SPACE)
    
    # 1. 查看当前Schema
    print("\n步骤1: 查看当前Schema...")
    resp = session.execute('DESCRIBE TAG player')
    if resp.is_succeeded():
        print("当前player标签的结构:")
        print(resp)
        
        # 检查embedding1是否存在
        has_embedding1 = False
        for i in range(resp.row_size()):
            field = resp.row_values(i)[0].as_string()
            if field == "embedding1":
                has_embedding1 = True
                print(f"发现embedding1字段，类型为: {resp.row_values(i)[1].as_string()}")
                break
    else:
        print("获取player标签结构失败:", resp.error_msg())
        sys.exit(1)
    
    # 2. 添加embedding1属性
    if not has_embedding1:
        print("\n步骤2: 添加新的embedding1属性...")
        resp = session.execute('ALTER TAG player ADD (embedding1 double)')
        if resp.is_succeeded():
            print("成功添加embedding1属性")
        else:
            print("添加embedding1属性失败:", resp.error_msg())
            sys.exit(1)
        
        # 等待Schema变更生效
        print("等待Schema变更生效...")
        time.sleep(10)
    else:
        print("\n步骤2: embedding1属性已存在，无需添加")
    
    # 3. 验证属性是否已添加
    print("\n步骤3: 验证embedding1属性是否已添加...")
    resp = session.execute('DESCRIBE TAG player')
    if resp.is_succeeded():
        print("更新后的player标签结构:")
        print(resp)
        
        # 再次检查embedding1是否存在
        has_embedding1 = False
        for i in range(resp.row_size()):
            field = resp.row_values(i)[0].as_string()
            if field == "embedding1":
                has_embedding1 = True
                print(f"确认embedding1字段已添加，类型为: {resp.row_values(i)[1].as_string()}")
                break
        
        if not has_embedding1:
            print("警告: 尽管操作成功，但embedding1字段未出现在Schema中")
            print("建议重启NebulaGraph服务或检查官方文档关于Schema变更的说明")
            sys.exit(1)
    else:
        print("获取player标签结构失败:", resp.error_msg())
        sys.exit(1)
    
    # 4. 获取所有球员ID
    print("\n步骤4: 获取所有球员ID...")
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
        print(f"找到 {len(player_ids)} 个球员")
    else:
        print("没有找到球员")
        sys.exit(1)
    
    # 5. 测试更新单个球员的embedding1
    print("\n步骤5: 测试更新单个球员的embedding1...")
    test_player = player_ids[0]
    test_value = 50.5  # 使用不同于embedding的值以区分
    test_query = f'UPSERT VERTEX ON player "{test_player}" SET embedding1 = {test_value}'
    resp = session.execute(test_query)
    
    if resp.is_succeeded():
        print(f"测试成功: 球员 {test_player} 的embedding1值已更新为 {test_value}")
        
        # 验证测试更新
        verify_query = f'FETCH PROP ON player "{test_player}" YIELD properties(vertex).name, properties(vertex).embedding, properties(vertex).embedding1'
        resp = session.execute(verify_query)
        if resp.is_succeeded():
            print("验证结果:")
            print(resp)
            
            # 6. 更新所有球员的embedding1
            print("\n步骤6: 更新所有球员的embedding1属性...")
            updated_count = 1  # 已经更新了一个测试球员
            
            for player_id in player_ids[1:]:  # 跳过第一个已测试的球员
                # 为embedding1生成40-50之间的随机值（与embedding不同的范围）
                random_value = round(np.random.uniform(40, 50), 2)
                upsert_query = f'UPSERT VERTEX ON player "{player_id}" SET embedding1 = {random_value}'
                resp = session.execute(upsert_query)
                
                if resp.is_succeeded():
                    updated_count += 1
                    # 每10个更新显示一次进度
                    if updated_count % 10 == 0 or updated_count == len(player_ids):
                        print(f"已更新 {updated_count}/{len(player_ids)} 个球员")
                else:
                    print(f"更新球员 {player_id} 失败: {resp.error_msg()}")
            
            print(f"\n总结: 成功更新 {updated_count}/{len(player_ids)} 个球员的embedding1属性")
            
            # 7. 验证最终结果
            print("\n步骤7: 验证最终结果...")
            if updated_count > 0:
                sample_ids = player_ids[:3]  # 取前3个球员作为样本
                sample_ids_str = '", "'.join(sample_ids)
                verify_query = f'FETCH PROP ON player "{sample_ids_str}" YIELD properties(vertex).name, properties(vertex).embedding, properties(vertex).embedding1'
                resp = session.execute(verify_query)
                
                if resp.is_succeeded():
                    print("验证几个球员的embedding和embedding1值:")
                    print(resp)
                else:
                    print("验证失败:", resp.error_msg())
        else:
            print(f"验证测试更新失败: {resp.error_msg()}")
    else:
        print(f"测试更新失败: {resp.error_msg()}")
        print("可能需要重启NebulaGraph服务或检查服务配置")

# 关闭连接池
connection_pool.close()
print("连接池已关闭")