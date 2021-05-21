# -*- coding: utf-8 -*-
# @project：wholee_keyword
# @author:caojinlei
# @file: test.py
# @time: 2021/05/09
# -*- coding: utf-8 -*-
# @project：wholee_keyword
# @author:caojinlei
# @file: bert_embedding.py
# @time: 2021/05/07
import torch
from transformers import BertTokenizer, BertModel, BertConfig

model_name = 'uncased_L-12_H-768_A-12'
tokenizer = BertTokenizer.from_pretrained(model_name)
model_config = BertConfig.from_pretrained(model_name)
model_config.output_hidden_states = True
model_config.output_attentions = True
bert_model = BertModel.from_pretrained(model_name, config=model_config)

# s = 'i have a pen'
s = 'i have a apple'
sen_code = tokenizer.encode(s)
print(sen_code)
sen_word = tokenizer.convert_ids_to_tokens(sen_code)
tokens_tensor = torch.LongTensor([sen_code])
segments_tensors = torch.zeros(len(sen_code), dtype=int)

# 静态词向量
emb = bert_model.embeddings.word_embeddings.weight.data
sen_emb = []
for i in sen_code:
    sen_emb.append(emb[i])

# 动态词向量
# bert_model.eval()
# with torch.no_grad():
#     outputs = bert_model(tokens_tensor, token_type_ids=segments_tensors)
#     encoded_layers = outputs  # outputs类型为tuple
#     print(encoded_layers[0].shape, encoded_layers[1].shape,
#           encoded_layers[2][0].shape, encoded_layers[3][0].shape)
#     m2=encoded_layers[0][0][3]
# 输出层为sequence_output, pooled_output, (hidden_states), (attentions)


# def DFS(graph, x, nodes_list, emb_list):  # 递归实现图的深度优先遍历
#     i = 0  # 若结点的相邻结点都被遍历,i回到上一个结点
#     for y in graph.neighbors(x):  # 结点的相邻结点遍历
#         i += 1
#         if y not in nodes_list:  # 如果此节点未被遍历,则加入list
#             nodes_list.append(y)
#             emb_list.append(graph.nodes[y]['emb'])
#             DFS(graph, y, nodes_list, emb_list)  # 递归,继续遍历
#         else:
#             if i == graph.degree(x):
#                 continue
#     return nodes_list, emb_list


G = nx.Graph()
G.add_node(1,emb=[1,2])
G.add_node(2,emb=[1,2])
G.add_node(3,emb=[1,2])
G.add_node(4,emb=[1,2])
G.add_node(5,emb=[1,2])
G.add_node(6,emb=[1,2])
G.add_node(7,emb=[1,2])
G.add_node(8,emb=[1,2])
G.add_node(9,emb=[1,2])
G.add_node(10,emb=[1,2])
G.add_node(11,emb=[1,2])
G.add_node(12,emb=[1,2])

link_node_list =[(1,2,1),
                 (1,3,1),
                 (4,1,1),
                 (5,2,1),
                 (6,3,1),
                 (7,4,1),
                 (8,7,1),
                 (9,8,1),
                 (10,11,1),
                 (11,12,1)
                 ]
G.add_weighted_edges_from(link_node_list)
# nodes =[1,2,3,4,5,6,7,8,9,10,11,12]
# graph_cluster_dict = {}
# for a in nodes:
#     x = nodes[0]
#     nodes_list = [x]
#     emb_list = [G.nodes[x]['emb']]
#     layer = 0
#     nodes_list, emb_list = DFS(G, x, nodes_list, emb_list, layer)
#     nodes = list(set(nodes).difference(set(nodes_list)))
#     G.remove_nodes_from(nodes_list)
#     graph_cluster_dict[x] = {'small_label': nodes_list, 'label_emb': emb_list, 'len': len(nodes_list)}
#     if not nodes:
#         break

['cotton','']
import heapq
import networkx as nx
import json
from utils import trans_word
import pandas as pd
import numpy as np
from Logginger import init_logger
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = init_logger('wholee_keyword', logging_path='output')
with open('output/word_dict/601_word_dict.json', 'r') as f:
    word_dict = json.load(f)

matrix_dis = np.load('output/matrix/601_cos_matrix_dis.npy')


nodes = list(word_dict.keys())
for a in nodes:
    x = nodes[0]
    nodes_list = [x]
    emb_list = [G.nodes[x]['emb']]
    layer = 0
    nodes_list, emb_list = DFS(G, x, nodes_list, emb_list,layer)
    nodes = list(set(nodes).difference(set(nodes_list)))
    G.remove_nodes_from(nodes_list)
    logger.info(f'还剩{len(nodes)}单词')
    graph_cluster_dict[x] = {'small_attr': nodes_list,'label_emb': emb_list, 'len': len(nodes_list)}
    if not nodes:
        break

name_search(word_dict,matrix_dis,'cotton',10)  # 面料
name_search(word_dict,matrix_dis,'spring',5)  # 季节
name_search(word_dict,matrix_dis,'sexy',9)  # 风格
name_search(word_dict,matrix_dis,'Loose',9)  # 版型

select_nodes = ['spring', 'shirt', 'loose', 'long', 'fashion', 'cotton', 'sexy']


def DFS(graph, x, nodes_list, emb_list, layer_list, layer, deep_layer, min_sim):  # 递归实现图的深度优先遍历
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
        if y not in nodes_list: # 如果此节点未被遍历,且和起始点相似度大于阈值，则加入list
            nodes_list.append(y)
            emb_list.append(graph.nodes[y]['emb'])
            layer_list.append(base_layer)
            DFS(graph, y, nodes_list, emb_list, layer_list, layer, deep_layer, min_sim)  # 递归,继续遍历
        else:
            if i == graph.degree(x):
                continue
    return nodes_list, emb_list, layer_list


word_dict = load_word_dict('601')
matrix_dis = load_sim_matrix('601', 'cos', word_dict)
G = create_graph(word_dict, matrix_dis,
                 0.5, 10, '601')

logger.info(f'深度遍历{label}图')
graph_cluster_dict = {}
if not select_nodes:
    logger.info(f'对中心词按数量排序')
    value_words = []
    for key, values in word_dict.items():
        value_words.append(values['count'])
    dictionary = dict(zip(key_words, value_words))
    max_words = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    select_nodes = [m for m, n in max_words]
    logger.info(f'针对全部单词进行无监督图聚类')
    for x in select_nodes:
        logger.info(f'还剩{len(select_nodes)}中心词')
        sim_list = [1]
        nodes_list = [x]
        layer_list = [0]
        layer = 1
        nodes_list, sim_list, layer_list = DFS(G, x, nodes_list, sim_list, layer_list, layer, deep_layer, min_sim)
        select_nodes = list(set(select_nodes).difference(set(nodes_list)))
        G.remove_nodes_from(nodes_list)
        graph_cluster_dict[x] = {'nodes_list': nodes_list, 'sim_list': sim_list, 'layer_list': layer_list,
                                 'len': len(nodes_list)}
        if not select_nodes:
            break


from nltk import po

r = "[0-9_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/\\\[\]'\"]"
for word in key_word:
    if bool(re.search(r, word)):
        del word_dict[word]
        continue
    if word_dict[word]['count'] > 50000 or word_dict[word]['count'] == 1:
        del word_dict[word]

from create_graph import *
from data_load import *
from utils import *
word_dict = load_word_dict('601')
matrix_dis = load_sim_matrix('601', 'cos', word_dict)
G = create_graph(word_dict, matrix_dis,
                 0.45, 10, '601')
select_nodes =['cotton']
re = graph_result(G, word_dict, matrix_dis,3, 0.4,select_nodes, '601')
trans_word(re['nodes_list'].values[0])
sim_fun(word_dict['golden']['emb'],word_dict['white']['emb'],'l1')

# 取交集
s1=['cotton', 'linen', 'wool','silk','denim','yarn','cloth','satin','leather','pu','skin','velvet']
s2 = ['velvet', 'leather', 'satin', 'silk','plastic', 'rubber','canvas','cloth','linen','microfiber','pu','cotton','wool','yarn','denim']
d = list(set(s1) | (set(s2)))

#判断是否存在
m1=['retro','european american','korean','japanese','college','ol','commuter','british','hippie','hiphop','bohemian','punk','simple','street','ethnic']
for i in m1:
    if word_dict.get(i):
        continue
    else:
        print(i)

['cotton', 'linen', 'wool','textile','silk','denim','yarn','cloth','satin','leather']
# 图案
['lattice','lace','stripe','plaid','leopard','dot','geometry','geometric']
格子/花边/条纹/格子/豹纹/圆点/几何/几何
['lattice','beige', 'wreath', 'simulate', 'sichuan', 'louvre', 'textile', 'cage']
颜色 beige
['white', 'black', 'red', 'blue', 'green', 'gray','beige','yellow','grey','purple','orange']
形容 ['large', 'small', 'huge', 'big', 'long', 'short','gigantic', 'oversized', 'wide', 'tiny', 'little','broad', 'narrow', 'widen']
# 工艺
['embroidery', 'stitch','knit']
['cixiu','pingjie','zhenzhi',]

# 特色 retro
[ 'vintage', 'retro','floral', 'elegant', 'graceful', 'exquisite', 'luxurious', 'fashionable', 'silky', 'sensual', 'glossy', 'plump', 'oversized', 'irregular', 'lantern', 'elderly', 'immortal', 'seaside', 'beautiful', 'gown', 'gothic', 'handsome', 'geometric', 'artistic', 'diagonal', 'thermal', 'legged', 'expensive', 'optional', 'colorful', 'generous', 'serpent', 'invisible', 'underwear', 'bohemian', 'explosive', 'faux', 'creative', 'vertical', 'shiny', 'reflective', 'romantic', 'fluffy', 'adjustable', 'playful', 'horizontal', 'conventional', 'curly', 'sanitary', 'wavy', 'technician', 'sensible', 'quilt']
#gudong/fugu/
# 种类 Windbreaker
['jeans', 'dress','pants', 'trouser', 'shorts',  'clothes', 'slacks', 'garments', 'skirts', 'robes', 'jacket',
 'Windbreaker','blouse','sweater','cloak','costume','shirt','robe']

# 风格
['retro','european american','korean','japanese','college','ol','commuter','british','hippie','hiphop','bohemian','punk','simple','street','ethnic']
复古风/欧美风/韩版/日系风/学院风/ol风格/通勤风/英伦风/嬉皮风/嘻哈风/波西米亚风/朋克风/简约风/街头风/民族风/

###612
['women', 'babies', 'men', 'sisters', 'people', 'boys', 'girlfriends', 'girls', 'ladies', 'teenagers', 'kids', 'children']
# 材料
['velvet', 'leather', 'satin', 'silk', 'crocodile','plastic', 'rubber','canvas','satin','cloth','linen','microfiber','pu','cotton','wool','yarn']

# 颜色
['yellow', 'blue', 'red', 'purple', 'white', 'beige', 'brown', 'green', 'gray', 'black', 'colorful', 'golden','colorful']

['velvet', 'leather', 'satin', 'silk','plastic', 'rubber','canvas','cloth','linen','microfiber','pu','cotton','wool','yarn','denim']
['cotton', 'linen', 'wool','silk','denim','yarn','cloth','satin','leather','pu','skin','velvet']

['low','mid-top','high']

# 特色
['cold', 'warm','non-slip','slack', 'thinner', 'crust', 'texture', 'stripe', 'cartoon','fashion', 'paragraph', 'scoop', 'bohemia', 'workout',
'extravagant','comfortable','lazy', 'elegant', 'fashionable', 'sexy','handsome', 'shiny','durable', 'exquisite', 'invisible', 'faux','sideways', 'usb',
 'skinny',  'widen','fray', 'portable', 'gorgeous', 'convenient', 'synthetic']

# 种类
shoes=['single', 'casual', 'sports', 'leather', 'canvas', 'running', 'cotton', 'cloth', 'board', 'flat', 'plate','slippers','loaf','martin', 'short', 'snow', 'rain', 'leather','heels']

