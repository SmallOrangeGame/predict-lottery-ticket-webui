# -*- coding:utf-8 -*-
"""
Author: BigCat
"""
import argparse
import requests
import lxml
import pandas as pd
from bs4 import BeautifulSoup
from loguru import logger
from config import os, name_path, data_save_name

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

parser = argparse.ArgumentParser()
parser.add_argument('--name', default="ssq", type=str, help="选择爬取数据: 双色球ssq/大乐透dlt")
args = parser.parse_args()


def get_current_number(name):
    """ 获取最新一期数字
    :return: int
    """
    url, path = get_url(name)
    url = "{}{}".format(url, "history.shtml")
    logger.info("访问【{}】最新一期数字 url = {}".format(name_path[name]["name"], url))
    r = requests.get(url=url, verify=False)
    r.encoding = "gb2312"
    soup = BeautifulSoup(r.text, "lxml")
    current_num = soup.find("div", class_="wrap_datachart").find("input", id="end")["value"]
    return current_num


def get_url(name):
    """
    :param name: 玩法名称
    :return:
    """
    url = "https://datachart.500.com/{}/history/".format(name)
    path = "newinc/{}.php?start={}&end="
    return url, path


def spider(name, start, end):
    """ 爬取历史数据
    :param name 玩法
    :param start 开始一期
    :param end 最近一期
    :return:
    """
    if name in name_path:
        data_list = dict()
        for history_name, history_data in name_path[name]["history"].items():
            url, path = get_url(name)
            url = "{}{}{}".format(url, path.format(history_name, start), end)
            logger.info("【{}】 第{}期到{}期 爬虫历史数据 url = {}".format(name_path[name]["name"], start, end, url))
            r = requests.get(url=url, verify=False)
            r.encoding = "gb2312"
            soup = BeautifulSoup(r.text, "lxml")
            tbody = soup.find("tbody", attrs={"id": "tdata"})
            if tbody is not None:
                trs = tbody.find_all("tr")
            else:
                trs = soup.find_all("tr", attrs={"class": "t_tr1"})
                if trs is None:
                    logger.warning("未找到符合条件的 tr 元素")
                    continue
            data = []
            for tr in trs:
                item = dict()
                for data_dic in history_data:
                    item[data_dic["tag"]] = tr.find_all("td")[data_dic["tr"]].get_text().strip()
                data.append(item)
                period = item[u"期数"]
                if period in data_list:
                    data_list[period].update(item)
                else:
                    data_list[period] = item
            df = pd.DataFrame(data)
            df = df.sort_values(by='期数')
            df.to_csv("{}{}".format(name_path[name]["path"], "{}.csv".format(history_name)), encoding="utf-8")
        # 数据汇总
        data_df = pd.DataFrame(list(data_list.values()))
        data_df.to_csv("{}{}".format(name_path[name]["path"], data_save_name), encoding="utf-8")
        return data_df
    else:
        logger.warning("抱歉，没有找到数据源！")


def run(name):
    current_number = get_current_number(name)
    logger.info("【{}】最新一期期号：{}".format(name_path[name]["name"], current_number))
    logger.info("正在获取【{}】数据。。。".format(name_path[name]["name"]))
    if not os.path.exists(name_path[name]["path"]):
        os.makedirs(name_path[name]["path"])
    data_list = spider(name, 1, current_number)
    if "data" in os.listdir(os.getcwd()):
        logger.info("【{}】 data.csv 数据准备就绪，共{}期, 下一步可训练模型...".format(name_path[name]["name"], len(data_list)))
    else:
        logger.error("数据文件不存在！")

if __name__ == '__main__':
    if not args.name:
        raise Exception("玩法名称不能为空！")
    else:
        run(name=args.name)
