Natural Language Inference任务介绍
NLI又叫做recognizing textual entailment. 是用来确定两句话是不是蕴含关系.

文本蕴含任务(text entailment)
它的任务形式是：给定一个前提文本（premise），根据这个前提去推断假说文本（hypothesis）与premise的关系，一般分为蕴含关系（entailment）和矛盾关系（contradiction），蕴含关系（entailment）表示从premise中可以推断出hypothesis；矛盾关系（contradiction）即hypothesis与premise矛盾。文本蕴含的结果就是这几个概率值。

Match-LSTM是由(Wang & Jiang,2016)发表在NAACL的论文提出，用于解决NLI(Natural Language Inference，文本蕴含)问题。

将premise和hypothesis两句话输入到两个LSTM中，用对应LSTM的隐层输出premise和hypothesis中每个位置对应上下文信息的一种表示，分别对应图中的HsHs和htht
对于hypothesis中的某个词的表示htihit与premise中每个词的表示HsHs计算得到一个权重向量，然后再对premise中的词进行加权求和，得到hypothesis的htihit在每个时刻对应的注意力向量aiai,即(attention过程)
把hypothesis中该词的表示htihit和其对应的attention的上下文向量aiai拼接在一起，输入到一个LSTM中，最后时刻的hmhm即hmNhNm作为输出以预测label。

# Reproducing Densely Interactive Inference Network in Keras

This repository aims to reproduce the results obtained in
[Natural Language Inference over Interaction Space](https://arxiv.org/abs/1709.04348) paper. <br/>
We've chosen this paper as a project for 
[reproducibility challenge](http://www.cs.mcgill.ca/~jpineau/ICLR2018-ReproducibilityChallenge.html) organized by ICLR. <br/>
Our reproducibility report with more details about experiments can be found here: https://arxiv.org/abs/1802.03198 <br/>
DIIN paper in OpenReview: https://openreview.net/forum?id=r1dHXnH6-&noteId=r1dHXnH6-


### 问题陈述(Problem statement)
文本蕴含问题：给定一个premise（前提），根据这个premise去判断相应的hypothesis（假说）正确与否，如果从这个premise中能够推断出这个hypothesis，那么就判断为entailment（蕴含），否则就是contradiction（矛盾）。
即给两个句子我们需要对这3个类别(`蕴含entailment`, `矛盾contradiction`, `中性neutral`)进行分类。
Given a premise sentence and a hypothesis one needs to determine whether hypothesis is
an entailment of the premise, a contradiction, or a neutral sentence. So given two sentences
we need to classify those between these 3 classes (`蕴含entailment`, `矛盾contradiction`, `中性neutral`).

* entailment(如果premise为真, 则hypothesis也必须为真)
* contradiction(如果premise为真, 则hypothesis必须为假)
* neutral(既不是entailment, 也不是contradiction)

下面介绍了MultiNLI数据集中的一些样本，这些样本是从DIIN论文中复制而来的。

`Premise`: The FCC has created two tiers of small business for this service with the approval of the SBA. <br/>
`Hypothesis`: The SBA has given the go-ahead for the FCC to divide this service into two tiers of small business. <br/>
`Label`: entailment.

`Premise`: He was crying like his mother had just walloped him. <br/>
`Hypothesis`: He was crying like his mother hit him with a spoon. <br/>
`Label`: Neutral


`Premise`: Later, Tom testified against John so as to avoid the electric chair. <br/>
`Hypothesis`: Tom refused to turn on his friend, even though he was slated to be executed. <br/>
`Label`: Contradiction


### 结构(Architecture)
`Encoding`, `Interaction`, `DecayingDropout` 在 `layers/` 中.
特征提取 ([DenseNet](https://arxiv.org/abs/1608.06993)) 在 `feature_extractors/` 中.
L2 optimizer 在 `optimizers/`.

### 说明(Instructions)

* 安装依赖 (`pip3` for `python3`)
```commandline
pip3 install -r requirements.txt
```

* 数据预处理
```commandline
python3 preprocess.py --p 32 --h 32 --chars_per_word 16 --save_dir data/ --dataset snli --word_vec_save_path data/word-vectors.npy

中文lcqmc数据集预处理(耗时约40min)
python3 lcqmc_preprocess.py --save_dir data/ --word_vec_save_path data/word-vectors.npy --train_data_dir /notebooks/data/LCQMC --word_vec_load_path /notebooks/data/Tencent_AILab_ChineseEmbedding.txt > preprocess.log

```

* 训练模型
```commandline
python train.py --batch_size 70 --eval_interval 500 --train_word_embeddings --char_embed_size 8 --char_conv_filters 100 --char_conv_kernel 5 --dropout_initial_keep_rate 1. --dropout_decay_rate 0.977 --dropout_decay_interval 10000 --first_scale_down_ratio 0.3 --transition_scale_down_ratio 0.5 --growth_rate 20 --layers_per_dense_block 8 --dense_blocks 3 --labels 3 --load_dir ./data --models_dir ./models/ --logdir ./logs --word_vec_path ./data/word-vectors.npy

中文lcqmc数据集训练(耗时约100min)
nohup python3 train.py --batch_size 24 --train_word_embeddings --load_dir ./data --models_dir ./models --logdir ./logs --word_vec_path ./data/word-vectors.npy > train.log &
```

* 在`TensorBoard`中查看结果
```commandline
tensorboard --logdir=./logs
```

* 测试
```
python3 predict.py 
```

### 结果(Results)
Currently we managed to obtain `87.34%` accuracy on `SNLI test` set, while the authors obtained `88.0%` accuracy on the same dataset. <br/>
Our best model can be downloaded from the first release v1.0: https://github.com/YerevaNN/DIIN-in-Keras/releases/download/v1.0/epoch.40-tloss.0.358379300162-tacc.0.873371346062.model

lcqmc dev_acc: 0.837



