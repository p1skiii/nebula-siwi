#!/usr/bin/env python
# -*- coding: utf-8 -*-
# filepath: /Users/wang/i/nebula-siwi/test.py
"""
此脚本用于探索NebulaGraph中的数据结构，并为player标签添加embedding属性。

第一步: 探索数据结构，查看player标签的定义和数据
第二步: 添加embedding属性
第三步: 为所有player实体生成和插入随机的embedding值

使用方法:
1. 确保已安装必要的依赖: pip install nebula3-python numpy
2. 根据实际情况修改以下配置变量
3. 运行脚本: python test.py
"""

import sys
import os
import time
import numpy as np
import json
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import ttypes

# NebulaGraph 连接配置
NEBULA_HOSTS = [('127.0.0.1', 9669)]  # 主机和端口
NEBULA_USER = 'root'                  # 用户名
NEBULA_PASSWORD = 'nebula'            # 密码
NEBULA_SPACE = 'basketballplayer'     # 图空间名

# Embedding 配置
EMBEDDING_DIM = 16                    # embedding维度，减少维度简化测试
SEED = 42                             # 随机数种子，确保可重现性

def create_connection_pool():
    """创建NebulaGraph连接池"""
    config = Config()
    config.max_connection_pool_size = 10
    pool = ConnectionPool()
    if not pool.init(NEBULA_HOSTS, config):
        raise RuntimeError("Failed to initialize connection pool")
    return pool

def explore_schema(session):
    """探索NebulaGraph中的Schema定义"""
    print("\n=== 探索 NebulaGraph Schema ===")
    
    # 1. 显示所有的标签(Tag)
    print("\n[1] 查看所有标签(Tags):")
    show_tags = session.execute("SHOW TAGS")
    if show_tags.is_succeeded():
        print(f"在图空间 '{NEBULA_SPACE}' 中发现 {show_tags.row_size()} 个标签:")
        for i in range(show_tags.row_size()):
            tag_name = show_tags.row_values(i)[0].as_string()
            print(f"  - {tag_name}")
    else:
        print(f"查询标签失败: {show_tags.error_msg()}")
    
    # 2. 查看player标签的详细定义
    print("\n[2] 查看player标签的详细定义:")
    desc_player = session.execute("DESC TAG player")
    if desc_player.is_succeeded():
        print("player标签的属性如下:")
        print(f"  共找到 {desc_player.row_size()} 个属性字段")
        
        # 打印列名
        if desc_player.row_size() > 0:
            print("\n  列信息:")
            column_names = []
            for i in range(desc_player.column_size()):
                column_names.append(desc_player.keys()[i])
            print(f"  列名: {column_names}")
            
        # 打印每个属性的详细信息
        print("\n  属性详细信息:")
        for i in range(desc_player.row_size()):
            row = desc_player.row_values(i)
            print(f"  属性 #{i+1}:")
            
            # 安全地获取每个字段，不假设具体的位置
            for j in range(len(row)):
                value_obj = row[j]
                value_type = value_obj.__class__.__name__
                
                # 根据对象类型安全获取值
                if value_obj.is_empty():
                    value = "空"
                elif value_obj.is_string():
                    value = f"'{value_obj.as_string()}'"
                elif value_obj.is_bool():
                    value = "是" if value_obj.as_bool() else "否"
                elif value_obj.is_int():
                    value = str(value_obj.as_int())
                elif value_obj.is_double():
                    value = str(value_obj.as_double())
                else:
                    value = f"<{value_type}类型>"
                
                column_name = desc_player.keys()[j] if j < len(desc_player.keys()) else f"列{j}"
                print(f"    {column_name}: {value}")
            
            # 特别查找字段名和类型（通常是第1列和第2列）
            field_name = row[0].as_string() if not row[0].is_empty() and row[0].is_string() else "未知"
            field_type = row[1].as_string() if len(row) > 1 and not row[1].is_empty() and row[1].is_string() else "未知"
            print(f"  -> 字段名: {field_name}, 类型: {field_type}")
            print("")
    else:
        print(f"查看player标签定义失败: {desc_player.error_msg()}")
    
    # 3. 获取player的示例数据
    print("\n[3] 获取player的示例数据:")
    sample_players = session.execute("MATCH (p:player) RETURN id(p), p.name, p.age LIMIT 5")
    if sample_players.is_succeeded():
        print(f"查询到 {sample_players.row_size()} 个player示例:")
        
        # 打印列信息
        print("  列信息:")
        for i in range(sample_players.column_size()):
            print(f"    列 #{i+1}: {sample_players.keys()[i]}")
            
        for i in range(sample_players.row_size()):
            row = sample_players.row_values(i)
            
            # 安全地获取ID
            player_id = None
            if not row[0].is_empty():
                if row[0].is_string():
                    player_id = row[0].as_string()
                elif row[0].is_int():
                    player_id = str(row[0].as_int())
                else:
                    player_id = f"<ID类型: {row[0].__class__.__name__}>"
            
            if player_id is None:
                print(f"  示例 #{i+1}: 无法获取ID")
                continue
                
            print(f"\n  player #{i+1}: ID = {player_id}")
            
            # 尝试获取名称和年龄
            name_value = "未知"
            age_value = "未知"
            
            if len(row) > 1 and not row[1].is_empty():
                if row[1].is_string():
                    name_value = row[1].as_string()
                    
            if len(row) > 2 and not row[2].is_empty():
                if row[2].is_int():
                    age_value = row[2].as_int()
                    
            print(f"    名称: {name_value}")
            print(f"    年龄: {age_value}")
            
            # 获取该player的所有属性
            try:
                all_props = session.execute(f'FETCH PROP ON player "{player_id}" YIELD properties(vertex)')
                if all_props.is_succeeded() and all_props.row_size() > 0:
                    print("    所有属性:")
                    for j in range(all_props.column_size()):
                        col_name = all_props.keys()[j]
                        value_obj = all_props.row_values(0)[j]
                        
                        if value_obj.is_empty():
                            print(f"      {col_name}: 空")
                        elif value_obj.is_map():
                            prop_map = value_obj.as_map()
                            print(f"      {col_name} (Map):")
                            for key, val in prop_map.items():
                                val_str = "空"
                                if not val.is_empty():
                                    if val.is_string():
                                        val_str = val.as_string()
                                    elif val.is_int():
                                        val_str = str(val.as_int())
                                    elif val.is_double():
                                        val_str = str(val.as_double())
                                    elif val.is_bool():
                                        val_str = "是" if val.as_bool() else "否"
                                    else:
                                        val_str = f"<{val.__class__.__name__}>"
                                print(f"        - {key}: {val_str}")
                        else:
                            print(f"      {col_name}: {value_obj}")
            except Exception as e:
                print(f"    获取属性时出错: {e}")
    else:
        print(f"查询player示例失败: {sample_players.error_msg()}")
    
    return True

def add_embedding_property(session):
    """为player标签添加embedding属性"""
    print("\n=== 添加embedding属性到player标签 ===")
    try:
        # 检查属性是否已存在
        result = session.execute(f"DESC TAG player")
        if not result.is_succeeded():
            print(f"获取player标签信息失败: {result.error_msg()}")
            return False
        
        # 检查结果中是否包含embedding属性
        has_embedding = False
        for i in range(result.row_size()):
            row = result.row_values(i)
            if row[0].is_string() and row[0].as_string() == "embedding":
                has_embedding = True
                print("player标签已有embedding属性，无需添加")
                break
        
        if not has_embedding:
            print("准备添加embedding属性 (为避免语法错误，将尝试不同的数据类型)...")
            
            # 获取NebulaGraph版本信息
            version_query = session.execute("RETURN '$^.0.0' AS ver")
            nebula_version = "unknown"
            if version_query.is_succeeded() and version_query.row_size() > 0:
                ver_value = version_query.row_values(0)[0]
                if ver_value.is_string():
                    nebula_version = ver_value.as_string()
            
            print(f"NebulaGraph版本信息: {nebula_version}")
            
            # 根据不同版本尝试不同的数据类型
            queries = []
            
            # 字符串类型（几乎所有版本都支持）
            queries.append("ALTER TAG player ADD (embedding string)")
            
            # 固定长度字符串（多数版本支持）
            queries.append("ALTER TAG player ADD (embedding FIXED_STRING(10000))")
            
            # 双精度类型（所有版本都支持）
            queries.append("ALTER TAG player ADD (embedding double)")
            
            # 列表类型尝试
            if nebula_version.startswith("3."):
                # NebulaGraph 3.x 可能支持的列表语法
                queries.append("ALTER TAG player ADD (embedding list<double>)")
            
            # 在NebulaGraph 2.x列表语法可能不同或不支持
            queries.append("ALTER TAG IF EXISTS player ADD (embedding list)")
            
            print(f"将尝试以下查询: {queries}")
            
            success = False
            for query in queries:
                print(f"\n尝试执行: {query}")
                alter_result = session.execute(query)
                if alter_result.is_succeeded():
                    success = True
                    print(f"成功添加embedding属性到player标签 (使用: {query})")
                    break
                else:
                    print(f"  尝试失败: {alter_result.error_msg()}")
            
            if not success:
                print("所有尝试都失败了。建议在NebulaGraph控制台手动添加embedding属性。")
                print("您可以尝试以下手动命令之一:")
                for query in queries:
                    print(f"  {query};")
                return False
            
            # 添加完属性后等待schema更新生效
            print("等待schema更新生效...")
            time.sleep(5)
        
        return True
    
    except Exception as e:
        print(f"添加embedding属性时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_all_player_ids(session):
    """获取所有player的ID"""
    print("获取所有player实体ID...")
    try:
        result = session.execute("MATCH (p:player) RETURN id(p) as player_id")
        if not result.is_succeeded():
            print(f"获取player ID失败: {result.error_msg()}")
            return []
        
        player_ids = []
        for i in range(result.row_size()):
            player_id = result.row_values(i)[0].as_string()
            player_ids.append(player_id)
        
        print(f"找到了{len(player_ids)}个player实体")
        return player_ids
    
    except Exception as e:
        print(f"获取player ID时发生错误: {e}")
        return []

def generate_random_embedding(dim=64, seed=None):
    """生成随机embedding向量"""
    if seed is not None:
        np.random.seed(seed + hash(time.time()) % 10000)  # 为每次生成添加一些随机性
    
    # 生成随机向量
    vec = np.random.normal(0, 1, dim)
    # 归一化向量
    vec = vec / np.linalg.norm(vec)
    # 转换为Python列表并保留4位小数
    return [round(float(x), 4) for x in vec]

def update_player_embeddings(session, player_ids):
    """为所有player更新embedding属性"""
    print("\n=== 为player实体添加embedding ===")
    
    # 首先检查player标签的结构，看embedding是什么类型
    desc_result = session.execute("DESC TAG player")
    embedding_type = None
    
    if desc_result.is_succeeded():
        for i in range(desc_result.row_size()):
            row = desc_result.row_values(i)
            field_name = row[0].as_string()
            if field_name == "embedding":
                embedding_type = row[1].as_string()
                print(f"发现embedding属性，类型为: {embedding_type}")
                break
    
    if not embedding_type:
        print("找不到embedding属性，请先添加该属性")
        return 0
    
    print(f"开始为player实体添加embedding (类型: {embedding_type})...")
    success_count = 0
    
    for i, player_id in enumerate(player_ids):
        try:
            # 为每个player生成一个唯一的embedding
            embedding_values = generate_random_embedding(EMBEDDING_DIM, SEED + i)
            
            # 根据不同的数据类型构造更新语句
            if "string" in embedding_type.lower() or "fixed_string" in embedding_type.lower():
                # 如果是字符串类型，将向量保存为JSON字符串
                embedding_json = json.dumps(embedding_values)
                update_query = f'UPDATE VERTEX "{player_id}" SET player.embedding = "{embedding_json}"'
            elif "double" in embedding_type.lower():
                # 如果是double类型，存储第一个值
                update_query = f'UPDATE VERTEX "{player_id}" SET player.embedding = {embedding_values[0]}'
                print(f"  警告: embedding类型为double，只能存储第一个值: {embedding_values[0]}")
            else:
                # 尝试直接用列表形式
                update_query = f'UPDATE VERTEX "{player_id}" SET player.embedding = {embedding_values}'
            
            print(f"  执行: {update_query}")
            result = session.execute(update_query)
            if result.is_succeeded():
                success_count += 1
                print(f"  成功更新player '{player_id}' 的embedding (进度: {success_count}/{len(player_ids)})")
            else:
                print(f"  更新player '{player_id}' 的embedding失败: {result.error_msg()}")
                print("  尝试备选方案...")
                
                # 备选方案：尝试不同的更新方式
                fallback_success = False
                
                # 尝试1：使用单引号而不是双引号
                if "string" in embedding_type.lower():
                    update_query = f"UPDATE VERTEX \"{player_id}\" SET player.embedding = '{embedding_json}'"
                    result = session.execute(update_query)
                    if result.is_succeeded():
                        fallback_success = True
                        success_count += 1
                        print(f"  备选方案1成功: {update_query}")
                
                # 尝试2：用字符串形式表示数组
                if not fallback_success:
                    embedding_str = str(embedding_values).replace("'", '"')
                    update_query = f'UPDATE VERTEX "{player_id}" SET player.embedding = "{embedding_str}"'
                    result = session.execute(update_query)
                    if result.is_succeeded():
                        fallback_success = True
                        success_count += 1
                        print(f"  备选方案2成功: {update_query}")
                
                if not fallback_success:
                    print(f"  所有更新尝试都失败了，跳过player '{player_id}'")
        
        except Exception as e:
            print(f"  更新player '{player_id}' 的embedding时发生错误: {e}")
    
    print(f"完成! 成功更新了 {success_count}/{len(player_ids)} 个player的embedding")
    return success_count

def verify_embeddings(session):
    """验证一些player是否已成功设置embedding"""
    print("\n=== 验证embedding设置情况 ===")
    
    # 获取embedding的类型信息
    desc_result = session.execute("DESC TAG player")
    embedding_type = None
    
    if desc_result.is_succeeded():
        for i in range(desc_result.row_size()):
            row = desc_result.row_values(i)
            field_name = row[0].as_string()
            if field_name == "embedding":
                embedding_type = row[1].as_string()
                print(f"embedding属性类型为: {embedding_type}")
                break
    
    # 随机选择几个player进行验证
    result = session.execute("MATCH (p:player) RETURN id(p) as player_id LIMIT 3")
    
    if not result.is_succeeded() or result.row_size() == 0:
        print("无法检索player进行验证")
        return
    
    for i in range(result.row_size()):
        player_id = result.row_values(i)[0].as_string()
        verify_query = f'FETCH PROP ON player "{player_id}" YIELD properties(vertex).embedding'
        
        verify_result = session.execute(verify_query)
        if verify_result.is_succeeded() and verify_result.row_size() > 0:
            embedding_value = verify_result.row_values(0)[0]
            
            print(f"\n验证player '{player_id}':")
            print(f"  - 值类型: {type(embedding_value)}")
            print(f"  - 是否为空: {embedding_value.is_empty()}")
            
            if embedding_value.is_empty():
                print(f"  验证失败: player '{player_id}' 的embedding未设置")
                continue
                
            # 根据不同的值类型尝试获取embedding
            if embedding_value.is_string():
                string_value = embedding_value.as_string()
                print(f"  - 字符串值: {string_value}")
                
                # 尝试解析JSON
                try:
                    embedding_list = json.loads(string_value)
                    if isinstance(embedding_list, list):
                        print(f"  验证成功: player '{player_id}' 的embedding是JSON格式的列表")
                        preview = embedding_list[:5]  # 只显示前5个元素
                        print(f"  embedding预览 (前5个值): {preview}...")
                        print(f"  embedding维度: {len(embedding_list)}")
                    else:
                        print(f"  字符串可以解析为JSON，但不是列表: {embedding_list}")
                except json.JSONDecodeError:
                    print(f"  字符串无法解析为JSON，可能是普通字符串")
            
            elif embedding_value.is_double():
                double_value = embedding_value.as_double()
                print(f"  - 浮点值: {double_value}")
                print(f"  验证成功: player '{player_id}' 的embedding是浮点数")
            
            elif embedding_value.is_list():
                try:
                    embedding_list = [val.as_double() for val in embedding_value.as_list()]
                    preview = embedding_list[:5]  # 只显示前5个元素
                    print(f"  验证成功: player '{player_id}' 的embedding是列表")
                    print(f"  embedding预览 (前5个值): {preview}...")
                    print(f"  embedding维度: {len(embedding_list)}")
                except Exception as e:
                    print(f"  无法将列表元素解析为浮点数: {e}")
            else:
                print(f"  验证: player '{player_id}' 的embedding类型未知")
                print(f"  值: {embedding_value}")
        else:
            error_msg = verify_result.error_msg() if verify_result else "未知错误"
            print(f"  验证查询失败: {error_msg}")

def main():
    """主函数"""
    print("开始探索NebulaGraph数据库结构并准备添加embedding属性")
    print(f"连接到NebulaGraph: {NEBULA_HOSTS}")
    
    pool = create_connection_pool()
    session = None
    
    try:
        session = pool.get_session(NEBULA_USER, NEBULA_PASSWORD)
        
        # 使用指定的图空间
        print(f"尝试切换到图空间: {NEBULA_SPACE}")
        use_result = session.execute(f"USE {NEBULA_SPACE}")
        if not use_result.is_succeeded():
            print(f"切换到图空间 {NEBULA_SPACE} 失败: {use_result.error_msg()}")
            
            # 列出所有可用的图空间
            show_spaces = session.execute("SHOW SPACES")
            if show_spaces.is_succeeded():
                print("\n可用的图空间包括:")
                for i in range(show_spaces.row_size()):
                    space_name = show_spaces.row_values(i)[0].as_string()
                    print(f"  - {space_name}")
                print("请修改脚本中的 NEBULA_SPACE 变量为上述图空间之一")
            return
        
        print(f"成功连接到图空间: {NEBULA_SPACE}")
        
        # 1. 探索数据库模式
        explore_schema(session)
        
        # 让用户确认是否继续
        user_input = input("\n是否继续添加embedding属性? (y/n): ")
        if user_input.lower() != 'y':
            print("操作已取消")
            return
            
        # 2. 添加embedding属性到player标签
        if not add_embedding_property(session):
            print("添加embedding属性失败，终止操作")
            return
            
        # 让用户确认是否继续添加随机embedding数据
        user_input = input("\n是否为player添加随机embedding数据? (y/n): ")
        if user_input.lower() != 'y':
            print("操作已取消")
            return
        
        # 3. 获取所有player的ID
        player_ids = get_all_player_ids(session)
        if not player_ids:
            print("没有找到player实体，终止操作")
            return
        
        # 4. 为所有player添加随机embedding
        update_player_embeddings(session, player_ids)
        
        # 5. 验证结果
        verify_embeddings(session)
        
        print("\n操作完成! player标签现在包含embedding属性，且所有player实体都已设置embedding值。")
    
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if session:
            session.release()
        pool.close()

if __name__ == "__main__":
    main()