# README
## 词性对照表
```ruby
CC  并列连词          NNS 名词复数        UH 感叹词
CD  基数词              NNP 专有名词        VB 动词原型
DT  限定符            NNP 专有名词复数    VBD 动词过去式
EX  存在词            PDT 前置限定词      VBG 动名词或现在分词
FW  外来词            POS 所有格结尾      VBN 动词过去分词
IN  介词或从属连词     PRP 人称代词        VBP 非第三人称单数的现在时
JJ  形容词            PRP$ 所有格代词     VBZ 第三人称单数的现在时
JJR 比较级的形容词     RB  副词            WDT 以wh开头的限定词
JJS 最高级的形容词     RBR 副词比较级      WP 以wh开头的代词
LS  列表项标记         RBS 副词最高级      WP$ 以wh开头的所有格代词
MD  情态动词           RP  小品词          WRB 以wh开头的副词
NN  名词单数           SYM 符号            TO  to
```
## 代码结构

```
├── Logginger.py
├── README.md
├── create_graph.py
├── data_load.py
├── input
│   ├── 601.csv  原始输入，直接从Zeppling中down下来，包含pid和title两列数据
│   ├── 601_trans.csv
├── output
│   ├── all.log 日志文件
│   ├── cluster_result 聚类结果
│   ├── error.log
│   ├── matrix  相似度矩阵
│   ├── operation_result  手动提取关键词结果
│   └── word_dict 词典
├── preprocess.py
├── run.py
├── test.py
├── uncased_L-12_H-768_A-12  BERT模型
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
└── utils.py
```

## 运行程序

```
python run.py  -s 1 -l 601
python run.py  -s 2 -l 601 e
```





