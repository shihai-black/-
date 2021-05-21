# -*- coding: utf-8 -*-
# @project：wholee_keyword
# @author:caojinlei
# @file: data_load.py
# @time: 2021/05/07
from transformers import BertTokenizer, BertModel, BertConfig
from Logginger import init_logger
import json
from utils import sim_matrix
import numpy as np

logger = init_logger('wholee_keyword', logging_path='output')


def load_bert_embedding(model_name):
    """
    载入Bert模型
    :param model_name:bert模型名
    :return:
    """
    model_name = model_name
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model_config = BertConfig.from_pretrained(model_name)
    model_config.output_hidden_states = True
    model_config.output_attentions = True
    bert_model = BertModel.from_pretrained(model_name, config=model_config)
    logger.info('载入模型成功')
    return tokenizer, bert_model


def load_word_dict(label):
    """
    载入处理好的数据
    :param label:具体类目id
    :return:
    """
    try:
        with open(f'output/word_dict/{label}_word_dict.json', 'r') as f:
            word_dict = json.load(f)
    except Exception as e:
        print(e)
    return word_dict


def load_sim_matrix(label, sim_name, word_dict):
    """
    载入相似度矩阵
    :param label:具体类目id
    :param sim_name:利用那种相似度
    :param word_dict:词典
    :return:
    """
    try:
        matrix_dis = np.load(f'output/matrix/{label}_{sim_name}_matrix_dis.npy')
    except Exception as e:
        print(e)
        matrix_dis = sim_matrix(sim_name, word_dict)
        np.save(f'output/matrix/{label}_{sim_name}_matrix_dis.npy', matrix_dis)
    return matrix_dis
