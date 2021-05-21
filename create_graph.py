# -*- coding: utf-8 -*-
# @project：wholee_keyword
# @author:caojinlei
# @file: create_graph.py
# @time: 2021/05/08
import networkx as nx
import pandas as pd
import copy
from Logginger import init_logger
from tqdm import tqdm

logger = init_logger('wholee_keyword', logging_path='output')


def create_graph(word_dict, matrix_dis, min_sim, max_edge, label):
    """
    构建底层图谱
    :param word_dict:词典
    :param matrix_dis:相似度矩阵
    :param min_sim:最小相似度
    :param max_edge:最多边
    :param label:类目id
    :return:
    """
    key_words = list(word_dict.keys())
    num_len = len(key_words)
    # 构建图
    logger.info(f'构建{label}底层图')
    G = nx.Graph()
    for i in range(num_len):
        G.add_node(key_words[i],
                   count=word_dict[key_words[i]]['count'],
                   tag=word_dict[key_words[i]]['tag'],
                   emb=word_dict[key_words[i]]['emb'])

    # 选出每个节点符合要求的点
    for i in tqdm(range(num_len)):
        sim_list = matrix_dis[i]
        base_node = key_words[i]
        base_tag = word_dict[base_node]['tag']
        link_node_list = []
        dictionary = dict(zip(key_words, sim_list))
        result_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        for node, dis in result_dict:
            link_tag = word_dict[base_node]['tag']
            if dis < min_sim or len(link_node_list) >= max_edge:
                break
            elif base_node == node:
                continue
            elif base_tag != link_tag:
                continue
            elif dis >= min_sim:
                link_node_list.append((base_node, node, dis))
        G.add_weighted_edges_from(link_node_list)
    return G


def graph_result(G, word_dict, matrix_dis, deep_layer, far_sim, select_nodes, label, to_excel=False):
    """
    底层边构建完成以后，对图谱实现图聚类和图搜索两个功能
    :param G:底层图谱
    :param word_dict:词典
    :param matrix_dis:相似度矩阵
    :param deep_layer:最深层数
    :param far_sim:最远节点与中心节点的最大相似度
    :param select_nodes: 选取节点【图搜索使用】
    :param label: 类目id
    :param to_excel: 是否需要输出结构
    :return:
    """
    key_words = list(word_dict.keys())

    # 递归实现图的深度优先遍历
    def DFS(graph, x, nodes_list, sim_list, layer_list, layer, deep_layer, far_sim):
        i = 0  # 若结点的相邻结点都被遍历,i回到上一个结点
        x_index = key_words.index(x)
        base_layer = layer
        for y in graph.neighbors(x):  # 结点的相邻结点遍历
            if base_layer > deep_layer:
                break
            i += 1
            layer += 1
            y_index = key_words.index(y)
            base2node_sim = matrix_dis[x_index, y_index]
            if y not in nodes_list:  # 如果此节点未被遍历,且和起始点相似度大于阈值，则加入list
                if base2node_sim < far_sim:
                    continue
                else:
                    nodes_list.append(y)
                    sim_list.append(base2node_sim)
                    layer_list.append(base_layer)
                    DFS(graph, y, nodes_list, sim_list, layer_list, layer, deep_layer, far_sim)  # 递归,继续遍历
            else:
                if i == graph.degree(x):
                    continue
        return nodes_list, sim_list, layer_list

    # 递归获取中心变量
    def reduce_center(x, G, select_nodes):
        sim_list = [1]
        nodes_list = [x]
        layer_list = [0]
        layer = 1
        nodes_list, sim_list, layer_list = DFS(G, x, nodes_list, sim_list, layer_list, layer, deep_layer, far_sim)
        select_nodes = list(set(select_nodes).difference(set(nodes_list)))
        G.remove_nodes_from(nodes_list)
        graph_cluster_dict[x] = {'nodes_list': nodes_list, 'sim_list': sim_list, 'layer_list': layer_list,
                                 'len': len(nodes_list)}
        return select_nodes, G, graph_cluster_dict

    # 图聚类
    logger.info(f'深度遍历{label}图')
    graph_cluster_dict = {}
    if not select_nodes:
        select_nodes = copy.deepcopy(key_words)
        logger.info(f'针对全部单词进行无监督图聚类')
        while True:
            if not select_nodes:
                break
            else:
                logger.info(f'还剩{len(select_nodes)}中心词')
                x = select_nodes[0]
                select_nodes, G, graph_cluster_dict = reduce_center(x, G, select_nodes)
    # 图搜索
    else:
        logger.info(f'针对部分单词进行无监督图搜')
        for x in select_nodes:
            logger.info(f'还剩{len(select_nodes)}中心词')
            sim_list = [1]
            nodes_list = [x]
            layer = 1
            layer_list = [0]
            nodes_list, sim_list, layer_list = DFS(G, x, nodes_list, sim_list, layer_list, layer, deep_layer, far_sim)
            graph_cluster_dict[x] = {'nodes_list': nodes_list, 'sim_list': sim_list, 'layer_list': layer_list,
                                     'len': len(nodes_list)}
            logger.info(f'还剩{len(select_nodes)}中心词')
            select_nodes.remove(x)
            if not select_nodes:
                break

    # 输出图聚类结果

    df_base = pd.DataFrame(graph_cluster_dict).T
    if to_excel:
        logger.info(f'输出{label}图聚类可视化结果')
        df_base.to_excel(f'output/cluster_result/{label}_graph_cluster.xlsx')
    return df_base
