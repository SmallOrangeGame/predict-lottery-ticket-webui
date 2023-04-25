# -*- coding:utf-8 -*-

import argparse
import json
import time
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
from config import *
from get_data import get_current_number, spider
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="dlt", type=str, help="选择训练数据: 双色球/大乐透")
args = parser.parse_args()

# 关闭eager模式
# Eager Execution 是 TensorFlow 2.x 中的一个特性，它允许在执行 TensorFlow 代码时立即计算结果，而不需要构建静态计算图。
# 但是，如果你希望使用 TensorFlow 1.x 的静态计算图模式，可以通过 disable_eager_execution() 方法来关闭 Eager Execution。
tf.compat.v1.disable_eager_execution()


def get_year():
    """ 截取年份
    eg：2020-->20, 2021-->21
    :return:
    """
    return int(str(datetime.datetime.now().year)[-2:])


def try_error(mode, name, predict_features, windows_size):
    """ 处理异常
    :param windows_size: 特征序列长度
    :param predict_features: 预测特征
    :param mode: 模式
    :param name: 玩法
    :return:predict_features
    """
    if mode:
        return predict_features
    else:
        if len(predict_features) != windows_size:
            logger.warning("期号出现跳期，期号不连续！开始查找最近上一期期号！本期预测时间较久！")
            last_current_year = (get_year() - 1) * 1000
            max_times = 160
            while len(predict_features) != 3:
                predict_features = spider(name, last_current_year + max_times, get_current_number(name), "predict")[[x[0] for x in ball_name]]
                time.sleep(np.random.random(1).tolist()[0])
                max_times -= 1
            return predict_features
        return predict_features


def get_ball_predict_result(predict_features, sequence_len, ball_args, windows_size):
    """ 获取预测结果
    """
    name_list = [(ball_name[ball_args], i + 1) for i in range(sequence_len)]
    data = predict_features[["{}_{}".format(name[0], i) for name, i in name_list]].values.astype(int) - 1
    # with red_graph.as_default():
    #     reverse_sequence = tf.compat.v1.get_default_graph().get_tensor_by_name(pred_key_d[ball_name[0][0]])
    #     pred = red_sess.run(reverse_sequence, feed_dict={
    #         "inputs:0": data.reshape(1, windows_size, sequence_len),
    #         "sequence_length:0": np.array([sequence_len] * 1)
    #     })
    # return pred, name_list


def get_final_result(name, predict_features, mode=0):
    """" 最终预测函数
    :param predict_features: 预测特征
    :param mode: 模式
    :param name: 玩法
    :return:predict_features
    """
    m_args = model_args[name]["model_args"]
    red_pred, red_name_list = get_ball_predict_result(predict_features=predict_features,
                                                      sequence_len=m_args["red"]["sequence_len"], ball_args="red",
                                                      windows_size=m_args["windows_size"])
    blue_pred, blue_name_list = get_ball_predict_result(predict_features=predict_features,
                                                      sequence_len=m_args["blue"]["sequence_len"], ball_args="blue",
                                                      windows_size=m_args["windows_size"])
    # ball_name_list = ["{}_{}".format(name[mode], i) for name, i in blue_name_list] + ["{}_{}".format(name[mode], i) for
    #                                                                                  name, i in blue_name_list]
    # pred_result_list = red_pred[0].tolist() + blue_pred[0].tolist()
    # return {
    #     b_name: int(res) + 1 for b_name, res in zip(ball_name_list, pred_result_list)
    # }


def load_ball_model(name, ball_args):
    """ 加载模型
    :param ball_args: 球名
    :param name: 玩法
    :return:
    """
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(
            "{}{}_model.ckpt.meta".format(model_args[args.name]["path"][ball_args], ball_args)
        )
    sess = tf.compat.v1.Session(graph=graph)
    saver.restore(sess, "{}{}_ball_model.ckpt".format(model_args[args.name]["path"][ball_args], ball_args))
    logger.info("已加载{}模型！".format(ball_name[ball_args]))


def load_all_model(name):
    # 加载关键节点名
    with open("{}/{}/{}".format(model_path, args.name, pred_key_name)) as f:
        pred_key_d = json.load(f)
    load_ball_model(name, ball_name["red"])
    load_ball_model(name, ball_name["blue"])


def run(name):
    windows_size = model_args[name]["model_args"]["windows_size"]
    current_number = get_current_number(args.name)
    logger.info("【{}】最近一期:{}".format(name_path[args.name]["name"], current_number))
    data = pd.read_csv("{}{}".format(name_path[name]["path"], data_save_name))
    if not len(data):
        raise logger.error(" 请执行 get_data.py 进行数据下载！")
    else:
        data = spider(name, 1, current_number)
    logger.info("【{}】预测期号：{}".format(name_path[name]["name"], int(current_number) + 1))
    predict_features = try_error(1, name, data.iloc[:windows_size], windows_size)
    logger.info("预测结果：{}".format(get_final_result(name, predict_features)))


if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(args.name)
