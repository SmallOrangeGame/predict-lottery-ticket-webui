# -*- coding:utf-8 -*-

import time
import json
import argparse
import numpy as np
import pandas as pd
from config import *
from modeling import LstmWithCRFModel, tf
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择训练数据: 双色球/大乐透")
args = parser.parse_args()

pred_key = {}


def create_train_data(name, windows):
    """ 创建训练数据
    :param name: 玩法，双色球/大乐透
    :param windows: 训练窗口
    :return:
    """
    data = pd.read_csv("{}{}".format(name_path[name]["path"], data_save_name))
    if not len(data):
        raise logger.error(" 请执行 get_data.py 进行数据下载！")
    else:
        # 创建模型文件夹
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        logger.info("【{}】训练数据已加载! ...".format(name_path[name]["name"]))
    data = data.iloc[:, 3:10].values
    logger.info("【{}】训练集数据维度: {}".format(name_path[name]["name"], data.shape))
    x_data, y_data = [], []
    sub_num = len(data) - windows - 1
    logger.info("【{}】训练数量 {}".format(name_path[name]["name"], sub_num))
    for i in range(sub_num):
        sub_data = data[i:(i + windows + 1), :]
        x_data.append(sub_data[1:])
        y_data.append(sub_data[0])
    return {
        "red": {
            "x_data": np.array(x_data)[:, :, :name_path[name]["cut_num"]],
            "y_data": np.array(y_data)[:, :name_path[name]["cut_num"]]
        },
        "blue": {
            "x_data": np.array(x_data)[:, :, name_path[name]["cut_num"]:],
            "y_data": np.array(y_data)[:, name_path[name]["cut_num"]:]
        }
    }


def train_ball_model(name, x_data, y_data, ball_args):
    """ 模型训练
    :param ball_args: 球名
    :param name: 玩法
    :param x_data: 训练样本
    :param y_data: 训练标签
    :return:
    """
    logger.info("开始训练【{}】{}模型...".format(name_path[name]["name"], ball_name[ball_args]))
    start_time = time.time()
    m_args = model_args[name]
    x_data = x_data - 1
    y_data = y_data - 1
    data_len = x_data.shape[0]
    logger.info("【{}】：【{}】 特征数据维度: {}".format(name_path[name]["name"], ball_name[ball_args], x_data.shape))
    logger.info("【{}】：【{}】 标签数据维度: {}".format(name_path[name]["name"], ball_name[ball_args], y_data.shape))
    with tf.compat.v1.Session() as sess:
        #  LSTM + CRF 解码模型
        ball_model = LstmWithCRFModel(
            # batch_size: 批处理大小，用于指定输入数据的批处理大小。
            batch_size=m_args["model_args"]["batch_size"],
            # w_size: 特征序列长度，用于指定输入数据中特征序列的长度。
            w_size=m_args["model_args"]["windows_size"],

            # n_class: 标签类别数量，用于指定模型输出的标签类别数量。
            n_class=m_args["model_args"][ball_args]["n_class"],
            # ball_num: 特征数量，用于指定输入数据中特征的数量。
            ball_num=m_args["model_args"][ball_args]["sequence_len"],
            # embedding_size: 嵌入层的维度，用于指定输入特征的嵌入层维度。
            embedding_size=m_args["model_args"][ball_args]["embedding_size"],
            # words_size: 词汇表大小，用于指定嵌入层中词汇表的大小。
            words_size=m_args["model_args"][ball_args]["n_class"],
            # hidden_size: LSTM 隐藏层的大小，用于指定 LSTM 隐藏层的大小。
            hidden_size=m_args["model_args"][ball_args]["hidden_size"],
            # layer_size: LSTM 层的数量，用于指定 LSTM 的层数。
            layer_size=m_args["model_args"][ball_args]["layer_size"]
        )
        # 定义一个优化器
        # AdamOptimizer 是一种常用的梯度下降优化算法
        train_step = tf.compat.v1.train.AdamOptimizer(
            # learning_rate（学习率）是梯度下降优化算法中控制参数更新步长的超参数。
            # 它决定了每一步参数更新的幅度大小，较大的学习率可能导致参数更新过大而无法收敛，较小的学习率可能导致参数更新过小而收敛速度较慢。
            learning_rate=m_args["train_args"][ball_args]["learning_rate"],
            # beta1 控制了一阶矩估计的衰减率，通常取较接近于 1 的值，例如 0.9
            beta1=m_args["train_args"][ball_args]["beta1"],
            # beta2 控制了二阶矩估计的衰减率，通常取较接近于 1 的值，例如 0.999。
            beta2=m_args["train_args"][ball_args]["beta2"],
            # epsilon 是 AdamOptimizer 中的一个数值稳定性的常数项，用于避免除零错误。
            # 它通常取一个较小的值，例如 1e-8，用于防止梯度更新时出现除零情况。
            epsilon=m_args["train_args"][ball_args]["epsilon"],
            # 控制是否使用锁定操作来保护共享变量的更新。如果设置为 True，那么在更新共享变量时会使用锁定操作，以确保在多线程环境下更新的安全性。
            # 如果设置为 False，那么将不会使用锁定操作，可能会导致在多线程环境下出现竞态条件。
            # 一般情况下，当在单线程环境下进行训练时，可以将 use_locking 设置为 False，以避免不必要的性能开销。
            use_locking=False,
            # name 是一个字符串参数，用于给优化器对象指定一个名称。
            name='Adam'
            # ball_model.loss 表示模型的损失，这个损失会被优化器最小化
            # minimize 方法会自动计算模型的梯度，并根据优化器的配置更新模型的参数。
        ).minimize(ball_model.loss)
        # 会在会话中执行全局变量初始化操作，并将所有全局变量的初始值设置为其初始值。
        sess.run(tf.compat.v1.global_variables_initializer())
        # 训练模型的循环
        for epoch in range(m_args["model_args"][ball_args]["epochs"]):
            for i in range(data_len):
                _, loss_, pred = sess.run(
                    [
                        # train_step: 是之前定义的优化器操作，用于更新模型的参数，使其朝着最小化损失函数的方向优化。
                        train_step,
                        # ball_model.loss: 是之前定义的模型的损失函数，用于评估模型在当前输入数据上的性能。
                        ball_model.loss,
                        # ball_model.pred_sequence: 是之前定义的模型的预测输出，表示模型在当前输入数据上的预测结果。
                        ball_model.pred_sequence
                    ],
                    # feed_dict: 是一个字典，用于将输入数据传入到模型的占位符（placeholders）
                    feed_dict={
                        # "inputs:0"对应模型定义中的self._inputs占位符，用于传入输入特征数据。
                        "inputs:0": x_data[i: (i + 1), :, :],
                        # "tag_indices:0"对应模型定义中的self._tag_indices占位符，用于传入标签数据。
                        "tag_indices:0": y_data[i: (i + 1), :],
                        # "sequence_length:0"对应模型定义中的self._sequence_length占位符，用于传入序列长度数据。
                        "sequence_length:0": np.array([m_args["model_args"][ball_args]["sequence_len"]] * 1),
                    },
                )
                if i % 100 == 0:
                    logger.info(
                        "epoch: {}, loss: {}, tag: {}, pred: {}".format(epoch, loss_, y_data[i: (i + 1), :][0] + 1,
                                                                        pred[0] + 1))
        pred_key[ball_args] = ball_model.pred_sequence.name
        if not os.path.exists(m_args["path"][ball_args]):
            os.makedirs(m_args["path"][ball_args])
        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "{}{}.{}".format(m_args["path"][ball_args], red_ball_model_name, extension), )
    logger.info("训练耗时: {}".format(time.time() - start_time))


def run(name):
    """ 执行训练
    :param name: 玩法
    :return:
    """
    logger.info("正在创建【{}】数据集...".format(name_path[name]["name"]))
    train_data = create_train_data(args.name, model_args[name]["model_args"]["windows_size"])

    train_ball_model(name=name, x_data=train_data["red"]["x_data"], y_data=train_data["red"]["y_data"],
                     ball_args="red")

    tf.compat.v1.reset_default_graph()  # 重置网络图

    train_ball_model(name=name, x_data=train_data["blue"]["x_data"], y_data=train_data["blue"]["y_data"],
                     ball_args="blue")

    # 保存预测关键结点名
    with open("{}/{}/{}".format(model_path, name, pred_key_name), "w") as f:
        json.dump(pred_key, f)


if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(args.name)
