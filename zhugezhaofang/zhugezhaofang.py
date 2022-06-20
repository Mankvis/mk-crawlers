# -*- coding: utf-8 -*-
# @Time    : 2022/6/20 11:17
# @Author  : Mankvis
# @Email   : chzzbeck@gmail.com
# @File    : zhugezhaofang.py
# @Software: PyCharm
import re
import requests


def unsbox(arg1):
    _0x4b082b = [0xf, 0x23, 0x1d, 0x18, 0x21, 0x10, 0x1, 0x26, 0xa, 0x9, 0x13, 0x1f, 0x28, 0x1b,
                 0x16, 0x17, 0x19, 0xd, 0x6, 0xb, 0x27, 0x12, 0x14, 0x8, 0xe, 0x15, 0x20, 0x1a,
                 0x2, 0x1e, 0x7, 0x4, 0x11, 0x5, 0x3, 0x1c, 0x22, 0x25, 0xc, 0x24]
    _0x4da0dc = []

    for i in _0x4b082b:
        _0x4da0dc.append(arg1[i - 1])
    _0x12605e = "".join(_0x4da0dc)
    return _0x12605e


def hexxor(s1, _0x4e08d8):
    _0x5a5d3b = ""

    for i in range(len(s1)):
        if i % 2 != 0:
            continue
        _0x401af1 = int(s1[i: i + 2], 16)
        _0x105f59 = int(_0x4e08d8[i: i + 2], 16)
        _0x189e2c_10 = (_0x401af1 ^ _0x105f59)
        _0x189e2c = hex(_0x189e2c_10)[2:]
        if len(_0x189e2c) == 1:
            _0x189e2c = '0' + _0x189e2c
        _0x5a5d3b += _0x189e2c
    return _0x5a5d3b


def index():
    # 首次请求获取 arg1
    url = "https://sh.esfxiaoqu.zhuge.com/page1/"
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)
    arg1 = re.search("var arg1='(.*?)';", response.text).group(1)

    # 把 arg1 带入计算 cookie
    _0x4e08d8 = "3000176000856006061501533003690027800375"
    s1 = unsbox(arg1)
    cookie = hexxor(s1, _0x4e08d8)
    headers.update({"cookie": f"acw_sc__v2={cookie}"})
    response = requests.get(url, headers=headers)
    print(response.text)


if __name__ == '__main__':
    index()
