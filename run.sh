#!/bin/bash

#nohup python3 train.py --batch_size 70 --eval_interval 500 --train_word_embeddings --char_embed_size 8 --char_conv_filters 100 --char_conv_kernel 5 --dropout_initial_keep_rate 1. --dropout_decay_rate 0.977 --dropout_decay_interval 10000 --first_scale_down_ratio 0.3 --transition_scale_down_ratio 0.5 --growth_rate 20 --layers_per_dense_block 8 --dense_blocks 3 --labels 3 --load_dir ./data --models_dir ./models2/ --logdir ./logs --word_vec_path ./data/word-vectors.npy > train2.log &

#nohup python3 lcqmc_preprocess.py --train_data_dir /home/gswyhq/data/LCQMC --word_vec_load_path /home/gswyhq/data/WordVector_60dimensional/wiki.zh.text.model > lcqmc_preprocess.log &
#nohup python3 lcqmc_preprocess.py --save_dir data/lcqmc_tencent/ --word_vec_save_path data/lcqmc-tencent-word-vectors.npy --train_data_dir /home/gswyhq/data/LCQMC --word_vec_load_path /home/gswyhq/data/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding.txt > lcqmc_preprocess_tencent_emb.log &

nohup python3 train.py --train_word_embeddings > train.log &

