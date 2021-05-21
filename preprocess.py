# -*- coding: utf-8 -*-
# @project：wholee_keyword
# @author:caojinlei
# @file: preprocess.py
# @time: 2021/05/07
"""
对数据进行预处理
1.获取每个单词的词向量和tag
2.对特别多或者少的词汇进行删除
"""
import json
from Logginger import init_logger
from nltk import pos_tag
from utils import *
import re
import pandas as pd

logger = init_logger('wholee_keyword', logging_path='output')


def get_word_attr(bert_model, tokenizer, label):
    """
    获取单词属性
    :param bert_model:bert模型
    :param tokenizer: 分词器
    :param label:类目标签id
    :return:
    """
    # 对数据进行预处理
    tagger = ['JJR', 'VBD', 'NNS', 'EX', 'JJ', 'UH', 'TO', 'CC', 'VB', 'RP', 'RB', 'VBP', 'RBR', 'WP', 'CD', 'RBS',
              'WDT', 'VBN', 'IN', '.', 'VBG', 'PRP$', 'DT', 'SYM', 'WRB', '``', 'NN', 'NNP', 'PRP', 'MD', 'FW', 'VBZ',
              'JJS', ':', 'POS', '#', "''", 'PDT', 'NNPS']
    r = "[0-9_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/\\\[\]'\"]"
    word_dict = {}
    line_count = 0
    index_emb = bert_model.embeddings.word_embeddings.weight.data  # 获取bert的静态词向量
    try:
        with open(f'input/{label}_trans.csv', 'r', encoding='utf8') as f:
            f.readline(1)
    except Exception as e:
        logger.info(e)
        csv_trans(f'input/{label}.csv')
        logger.info(f'get input/{label}_trans.csv')
    with open(f'input/{label}_trans.csv', 'r', encoding='utf8') as f:
        for lines in f.readlines():
            line_count += 1
            lines = lines.strip()
            title = lines.split(';')[1].lower()  # 单词转化小写
            lines = re.sub(r, '', title)
            sen_code = tokenizer.encode(lines)  # 数字化
            sen_word = tokenizer.convert_ids_to_tokens(sen_code)  # id转化
            tag_result = pos_tag(sen_word)  # 词性标注
            for i in range(len(sen_code)):
                tag = tag_result[i][1]
                tag_index = tagger.index(tag)
                word = tag_result[i][0]
                if word_dict.get(word):
                    word_dict[word]['count'] += 1
                    word_dict[word]['tag_list'][tag_index] += 1
                else:
                    index = sen_code[i]
                    emb = index_emb[index].tolist()
                    tag_list = [0 for _ in range(len(tagger))]
                    tag_list[tag_index] = 1
                    word_dict[word] = {'count': 1, 'emb': emb, 'tag_list': tag_list}
            if line_count % 5000 == 0:
                logger.info(f'this txt has finish {line_count} line')
    # 输出原始数据
    with open(f'output/word_dict/{label}_origin_word_dict.json', 'w') as f:
        json.dump(word_dict, f)
    logger.info(f'单词字典原始输出完成,单词总数量:{len(word_dict)}')

    # 对数据按count进行排序
    key_word = list(word_dict.keys())
    value_words = []
    for key, values in word_dict.items():
        value_words.append(values['count'])
    dictionary = dict(zip(key_word, value_words))
    max_words = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    refresh_word = [m for m, n in max_words]

    # 去除value_count<=10且前2的单词，还有key含有数字,特殊符号。
    new_word_dict = {}
    for word in refresh_word[2:]:
        if bool(re.search(r, word)):
            continue
        if len(word) <= 2:
            continue
        if word_dict[word]['count'] <= 10:
            break
        else:
            new_word_dict[word] = {'count': word_dict[word]['count'],
                                   'emb': word_dict[word]['emb'],
                                   'tag_list': word_dict[word]['tag_list']}
            tag_list = new_word_dict[word]['tag_list']
            new_word_dict[word]['tag'] = tagger[tag_list.index(max(tag_list))]

    # 输出删除后的数据
    logger.info('单词预处理完成')
    with open(f'output/word_dict/{label}_word_dict.json', 'w') as f:
        json.dump(new_word_dict, f)
    logger.info(f'单词字典预处理输出完成:单词总数量{len(new_word_dict)}')

    # 输出可视化数据
    logger.info('输出单词可视化结果')
    df_base = pd.DataFrame(new_word_dict).T.drop(columns=['emb', 'tag_list'])
    df_base.to_excel(f'output/word_dict/{label}_word.xlsx')
    return df_base
