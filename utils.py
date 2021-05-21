# -*- coding: utf-8 -*-
# @project：wholee_keyword
# @author:caojinlei
# @file: utils.py
# @time: 2021/05/08
import numpy as np
from scipy.stats import pearsonr
from translate import Translator
from tqdm import tqdm
import heapq
import pandas as pd


def sim_fun(x, y, name):
    """
    相似度比较函数
    :param x:向量
    :param y:向量
    :param name:具体相似度函数名称
    :return:
    """
    if name == 'l2':
        result = np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))
    elif name == 'l1':
        result = sum(abs(a - b) for a, b in zip(x, y))
    elif name == 'cos':
        x_norm = np.linalg.norm(np.array(x))
        y_norm = np.linalg.norm(np.array(y))
        result = np.dot(x, y) / (x_norm * y_norm)
    else:
        result = pearsonr(x, y)[0]
    return result


def sim_matrix(name, word_dict):
    """
    相似度矩阵
    :param name:具体相似度函数名称
    :param word_dict:单词详细信息字典
    :return:
    """
    key_word = list(word_dict.keys())
    dict_len = len(word_dict)
    matrix_dis = np.zeros(shape=(dict_len, dict_len))
    for i in tqdm(range(dict_len)):
        emb_base = word_dict[key_word[i]]['emb']
        tag_base = word_dict[key_word[i]]['tag']
        for j in range(dict_len):
            tag_compare = word_dict[key_word[j]]['tag']
            if tag_base != tag_compare:
                sim = -9999
            else:
                emb_compare = word_dict[key_word[j]]['emb']
                sim = sim_fun(emb_base, emb_compare, name)
            matrix_dis[i][j] = sim
    return matrix_dis


def trans_word(word_list):
    """
    翻译工具
    :param word_list:英文单词列表
    :return:中文单词列表
    """
    translator = Translator(to_lang="chinese")
    chinese_list = []
    for word in word_list:
        chinese_list.append(translator.translate(word))
    return chinese_list


def name_search(word_dict, matrix_dis, name, n):
    """
    按word搜索单词最相似的前top_n单词
    >>>name_search(word_dict,matrix_dis,'cotton',10)  # 面料
    :param word_dict:单词详细信息字典
    :param matrix_dis:相似度矩阵
    :param name:单词名
    :param n:top_n
    :return:
    """
    key_words = list(word_dict.keys())
    word_index = key_words.index(name)
    con = matrix_dis[word_index].tolist()
    max_number = heapq.nlargest(n, con)
    max_index = []
    name_list = []
    for t in max_number:
        index = con.index(t)
        max_index.append(index)
        name_list.append(key_words[index])
        con[index] = 0

    return max_number, max_index, name_list


def compare_name(word1, word2, word_dict, matrix_dis):
    """
    比较两个单词相似度
    :param word1:
    :param word2:
    :param word_dict:
    :param matrix_dis:
    :return:
    """
    key_words = list(word_dict.keys())
    word1_index = key_words.index(word1)
    word2_index = key_words.index(word2)
    return matrix_dis[word1_index][word2_index]


def get_count(word_dict, name_list):
    """
    获取每句话单词在全局的数量
    :param word_dict:词典
    :param name_list:语句
    :return:
    """
    count_list = []
    for name in name_list:
        count_list.append(word_dict[name]['count'])
    return count_list


def csv_trans(input_path):
    """
    由于Zeppling上下载的数据是按','作为分隔符，然后title中也存在','，因此将csv转换为';'的csv
    :param input_csv:输入csv路径
    :return:
    """
    df = pd.read_csv(input_path)
    input_trans = list(input_path)
    input_trans.insert(-4, '_trans')
    output_csv = ''.join(input_trans)
    df.to_csv(output_csv, sep=';', index=None, header=None)
