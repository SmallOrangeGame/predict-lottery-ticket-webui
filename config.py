# -*- coding: utf-8 -*-

import os

# 模型名
model_path = os.getcwd() + "/model/"
pred_key_name = "key_name.json"
red_ball_model_name = "red_ball_model"
blue_ball_model_name = "blue_ball_model"
extension = "ckpt"

data_save_name = "data.csv"

ball_name = {
    "red": "红球",
    "blue": "篮球",
}

name_path = {
    "ssq": {
        "name": "双色球",
        "path": "data/ssq/",
        "cut_num": 6,
        "history": {
            "outball": [
                {"tag": u"期数", "tr": 0},
                {"tag": u"开奖日期", "tr": 1},
                {"tag": u"红球_1", "tr": 8},
                {"tag": u"红球_2", "tr": 9},
                {"tag": u"红球_3", "tr": 10},
                {"tag": u"红球_4", "tr": 11},
                {"tag": u"红球_5", "tr": 12},
                {"tag": u"红球_6", "tr": 13},
                {"tag": u"蓝球_1", "tr": 14},
                {"tag": u"红球_1 顺序", "tr": 2},
                {"tag": u"红球_2 顺序", "tr": 3},
                {"tag": u"红球_3 顺序", "tr": 4},
                {"tag": u"红球_4 顺序", "tr": 5},
                {"tag": u"红球_5 顺序", "tr": 6},
                {"tag": u"红球_6 顺序", "tr": 7},
            ],
            "history": [
                {"tag": u"开奖日期", "tr": 15},
                {"tag": u"期数", "tr": 0},
                {"tag": u"红球_1", "tr": 1},
                {"tag": u"红球_2", "tr": 2},
                {"tag": u"红球_3", "tr": 3},
                {"tag": u"红球_4", "tr": 4},
                {"tag": u"红球_5", "tr": 5},
                {"tag": u"红球_6", "tr": 6},
                {"tag": u"蓝球_1", "tr": 7},
                {"tag": u"奖池奖金", "tr": 9},
                {"tag": u"一等奖注数", "tr": 10},
                {"tag": u"一等奖奖金", "tr": 11},
                {"tag": u"二等奖注数", "tr": 12},
                {"tag": u"二等奖奖金", "tr": 13},
                {"tag": u"总投注额", "tr": 14},
            ],
            "history_same": [
                {"tag": u"期数", "tr": 1},
                {"tag": u"红球_1", "tr": 2},
                {"tag": u"红球_2", "tr": 3},
                {"tag": u"红球_3", "tr": 4},
                {"tag": u"红球_4", "tr": 5},
                {"tag": u"红球_5", "tr": 6},
                {"tag": u"红球_6", "tr": 7},
                {"tag": u"蓝球_1", "tr": 8},
                {"tag": u"和值", "tr": 9},
                {"tag": u"平均值", "tr": 10},
                {"tag": u"尾数和值", "tr": 11},
                {"tag": u"奇号个数", "tr": 12},
                {"tag": u"偶号个数", "tr": 13},
                {"tag": u"奇偶偏差", "tr": 14},
                {"tag": u"奇号连续", "tr": 15},
                {"tag": u"偶号连续", "tr": 16},
                {"tag": u"大号个数", "tr": 17},
                {"tag": u"小号个数", "tr": 18},
                {"tag": u"大小偏差", "tr": 19},
                {"tag": u"尾号组数", "tr": 20},
                {"tag": u"AC值", "tr": 21},
                {"tag": u"连号个数", "tr": 22},
                {"tag": u"连号组数", "tr": 23},
                {"tag": u"首尾差", "tr": 24},
                {"tag": u"最大间距", "tr": 25},
                {"tag": u"同位相同", "tr": 26},
                {"tag": u"重号个数", "tr": 27},
                {"tag": u"斜号个数", "tr": 28},
            ],
            # "back_prize": [
            #     {"tag": u"期数", "tr": 0},
            #     {"tag": u"开奖日期", "tr": 1},
            #     {"tag": u"一等奖注数", "tr": 3},
            #     {"tag": u"一等奖奖金", "tr": 4},
            #     {"tag": u"二等奖注数", "tr": 5},
            #     {"tag": u"二等奖奖金", "tr": 6},
            #     {"tag": u"六等奖注数", "tr": 7},
            #     {"tag": u"本期投注", "tr": 8},
            #     {"tag": u"返奖金额", "tr": 9},
            #     {"tag": u"返奖比例", "tr": 10},
            #     {"tag": u"奖池金额", "tr": 11},
            # ]
        },
    },
    "dlt": {
        "name": "大乐透",
        "path": "data/dlt/",
        "cut_num": 5,
        "history": {
            "outball": [
                {"tag": u"期数", "tr": 0},
                {"tag": u"开奖日期", "tr": 1},
                {"tag": u"红球_1", "tr": 7},
                {"tag": u"红球_2", "tr": 8},
                {"tag": u"红球_3", "tr": 9},
                {"tag": u"红球_4", "tr": 10},
                {"tag": u"红球_5", "tr": 11},
                {"tag": u"蓝球_1", "tr": 12},
                {"tag": u"蓝球_2", "tr": 13},
                {"tag": u"红球_1 顺序", "tr": 2},
                {"tag": u"红球_2 顺序", "tr": 3},
                {"tag": u"红球_3 顺序", "tr": 4},
                {"tag": u"红球_4 顺序", "tr": 5},
                {"tag": u"红球_5 顺序", "tr": 6},
            ],
            "history": [
                {"tag": u"期数", "tr": 0},
                {"tag": u"开奖日期", "tr": 14},
                {"tag": u"红球_1", "tr": 1},
                {"tag": u"红球_2", "tr": 2},
                {"tag": u"红球_3", "tr": 3},
                {"tag": u"红球_4", "tr": 4},
                {"tag": u"红球_5", "tr": 5},
                {"tag": u"蓝球_1", "tr": 6},
                {"tag": u"蓝球_2", "tr": 7},
                {"tag": u"奖池奖金", "tr": 8},
                {"tag": u"一等奖注数", "tr": 9},
                {"tag": u"一等奖奖金", "tr": 10},
                {"tag": u"二等奖注数", "tr": 11},
                {"tag": u"二等奖奖金", "tr": 12},
                {"tag": u"总投注额", "tr": 13},
            ],

            "history_same": [
                {"tag": u"期数", "tr": 0},
                {"tag": u"红球_1", "tr": 1},
                {"tag": u"红球_2", "tr": 2},
                {"tag": u"红球_3", "tr": 3},
                {"tag": u"红球_4", "tr": 4},
                {"tag": u"红球_5", "tr": 5},
                {"tag": u"蓝球_1", "tr": 6},
                {"tag": u"蓝球_2", "tr": 7},
                {"tag": u"和值", "tr": 8},
                {"tag": u"平均值", "tr": 9},
                {"tag": u"尾数和值", "tr": 10},
                {"tag": u"奇号个数", "tr": 11},
                {"tag": u"偶号个数", "tr": 12},
                {"tag": u"奇偶偏差", "tr": 13},
                {"tag": u"奇号连续", "tr": 14},
                {"tag": u"偶号连续", "tr": 15},
                {"tag": u"大号个数", "tr": 16},
                {"tag": u"小号个数", "tr": 17},
                {"tag": u"大小偏差", "tr": 18},
                {"tag": u"尾号组数", "tr": 19},
                {"tag": u"AC值", "tr": 20},
                {"tag": u"连号个数", "tr": 21},
                {"tag": u"连号组数", "tr": 22},
                {"tag": u"首尾差", "tr": 23},
                {"tag": u"最大间距", "tr": 24},
                {"tag": u"同位相同", "tr": 25},
                {"tag": u"重号个数", "tr": 26},
                {"tag": u"斜号个数", "tr": 27},
            ],
        },
    }
}

model_args = {
    "ssq": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red": {
                "sequence_len": 6,
                "n_class": 33,
                "epochs": 1,
                "embedding_size": 32,
                "hidden_size": 32,
                "layer_size": 1,
            },
            "blue": {
                "sequence_len": 1,
                "n_class": 16,
                "epochs": 1,
                "embedding_size": 32,
                "hidden_size": 32,
                "layer_size": 1,
            },
        },
        "train_args": {
            "red": {
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08,
                },
            "blue": {
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08,
            },
        },
        "path": {
            "red": model_path + "/ssq/red_ball_model/",
            "blue": model_path + "/ssq/blue_ball_model/"
        }
    },
    "dlt": {
        "model_args": {
            "windows_size": 3,
            "batch_size": 1,
            "red": {
                "sequence_len": 5,
                "n_class": 36,
                "epochs": 1,
                "embedding_size": 32,
                "hidden_size": 32,
                "layer_size": 1,
            },
            "blue": {
                "sequence_len": 2,
                "n_class": 13,
                "epochs": 1,
                "embedding_size": 32,
                "hidden_size": 32,
                "layer_size": 1,
            },
        },
        "train_args": {
            "red": {
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08,
                },
            "blue": {
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08,
            },
        },
        "path": {
            "red": model_path + "/dlt/red_ball_model/",
            "blue": model_path + "/dlt/blue_ball_model/"
        }
    }
}
