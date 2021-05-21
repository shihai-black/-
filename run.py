# -*- coding: utf-8 -*-
# @project：wholee_keyword
# @author:caojinlei
# @file: run.py
# @time: 2021/05/09
import argparse
from data_load import *
from preprocess import get_word_attr
from create_graph import create_graph, graph_result


def argments():
    """
    外部可配参数
    """
    parser = argparse.ArgumentParser(description='Human space-time domain')
    parser.add_argument('-mn', '--model_name', type=str, default='uncased_L-12_H-768_A-12',
                        help='Bert model name')
    parser.add_argument('-l', '--label', type=str, default='601',
                        help='label name')
    parser.add_argument('-sn', '--sim_name', type=str, default='cos',
                        help='Similarity function')
    parser.add_argument('-d', '--deep_layer', type=int, default=3, metavar='N',
                        help='The deepest layer of a graph (default: 3)')
    parser.add_argument('-min', '--min_sim', type=float, default=0.5, metavar='N',
                        help='Similarity threshold (default: 0.5)')
    parser.add_argument('-fs', '--far_sim', type=float, default=0.45, metavar='N',
                        help='Farthest similarity threshold (default: 0.45)')
    parser.add_argument('-me', '--max_edge', type=int, default=10, metavar='N',
                        help='Maximum number of nodes (default: 10)')
    parser.add_argument('-s', '--step', type=str, default='1',
                        help='Step 1 represents data preprocessing and Step 2 represents graph generation clustering ')
    parser.add_argument('-e', '--to_excel', action='store_true', default=False,
                        help='Whether to excel the result')
    return parser.parse_args()


def cmd_entry(args, select_nodes=None):
    """
    Step1:载入词向量，对数据进行预处理
    Step2：对输出可视化数据集合进行观察，挑选词汇
    Step3：生成图，对挑选的词汇进行初步聚类
    Step4：人工介入，对聚类后的结果进行选出
    Step5【待续】：匹配回原先的词语（可以考虑后续和pid碰撞的时候以词根进行匹配）
    :return:
    """
    if args.step == '1':
        tokenizer, bert_model = load_bert_embedding(args.model_name)
        re = get_word_attr(bert_model, tokenizer, label=args.label)
    else:
        word_dict = load_word_dict(args.label)
        matrix_dis = load_sim_matrix(args.label, args.sim_name, word_dict)
        G = create_graph(word_dict, matrix_dis,
                         args.min_sim, args.max_edge, args.label)
        re = graph_result(G, word_dict, matrix_dis,
                          args.deep_layer, args.far_sim, select_nodes, args.label,
                          to_excel=args.to_excel)
    return re


if __name__ == '__main__':
    args = argments()
    print(args)
    df = cmd_entry(args=args)
