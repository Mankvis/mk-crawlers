# -*- coding: utf-8 -*-
# @Time    : 2022/6/16 14:56
# @Author  : Mankvis
# @Email   : chzzbeck@gmail.com
# @File    : slide.py
# @Software: PyCharm

"""
没有爬虫部分,只是一个安居客滑块的分析,爬虫直接发挥
校验失败可以在轨迹处下下功夫
"""

import base64
import json
import re
from typing import Union
from urllib.parse import quote_plus

import cv2
import numpy as np
import requests
from Crypto.Cipher import AES as _AES
from Crypto.Util.Padding import pad, unpad
from loguru import logger
from requests import Session


class GapGeneral(object):
    """ https://www.cnblogs.com/yoyoketang/p/14731542.html """

    def __init__(self, bg_resize=None, sid_resize=None):
        self.bg_resize = bg_resize
        self.sid_resize = sid_resize

    def _download_im(self, url: str) -> bytes:
        return requests.get(url).content

    def _load_im(self, im: Union[str, bytes], im_type: str) -> np.ndarray:
        if im.startswith('https') or im.startswith('http'):
            im = self._download_im(im)
            im = np.frombuffer(im, np.uint8)
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        if isinstance(im, str):
            im = cv2.imread(im)
        if isinstance(im, bytes):
            im = np.frombuffer(im, np.uint8)
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
        if im_type == 'bg' and self.bg_resize:
            im = cv2.resize(im, self.bg_resize)
        if im_type == 'sid' and self.sid_resize:
            im = cv2.resize(im, self.sid_resize)
        return im

    def _im2gray(self, im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    def get_distance(self, bg_im: Union[str, bytes], sid_im: Union[str, bytes]) -> int:
        # 加载图片,支持本地绝对路径、bytes、https/http地址
        bg_im = self._load_im(bg_im, 'bg')
        sid_im = self._load_im(sid_im, 'sid')
        # 清除滑块周围空白区域
        sid_im = self.clear_white(sid_im)
        sid_im = cv2.cvtColor(sid_im, cv2.COLOR_RGB2GRAY)
        # 图片边缘化处理
        bg_im = self._canny(bg_im)
        sid_im = self._canny(sid_im)

        # 图片灰度处理
        bg_im = self._im2gray(bg_im)
        sid_im = self._im2gray(sid_im)
        # 图片高斯滤波处理
        # convert_bg_im = self._gaussian_blur(bg_im)
        # convert_sid_im = self._gaussian_blur(sid_im)

        # # 模版匹配
        distance = self.template_match(sid_im, bg_im)
        return distance

    def _gaussian_blur(self, im: np.ndarray) -> np.ndarray:
        kernel = (3, 3)
        std = 0
        return cv2.GaussianBlur(im, kernel, std)

    def _canny(self, im: np.ndarray) -> np.ndarray:
        return cv2.Canny(im, 100, 200)

    def _match(self, bg_im: np.ndarray, sid_im: np.ndarray) -> int:
        result = cv2.matchTemplate(bg_im, sid_im, cv2.TM_CCOEFF_NORMED)
        # 最小值，最大值，并得到最小值, 最大值的索引
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        # 横坐标
        distance = max_loc[0]
        # 展示圈出来的区域
        x, y = max_loc  # 获取x,y位置坐标
        w, h = sid_im.shape[::-1]  # 宽高
        cv2.rectangle(bg_im, (x, y), (x + w, y + h), (7, 249, 151), 2)
        return distance

    def template_match(self, tpl, target):
        th, tw = tpl.shape[:2]
        result = cv2.matchTemplate(target, tpl, cv2.TM_CCOEFF_NORMED)
        # 寻找矩阵(一维数组当作向量,用Mat定义) 中最小值和最大值的位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        tl = max_loc
        br = (tl[0] + tw, tl[1] + th)
        # 绘制矩形边框，将匹配区域标注出来
        # target：目标图像
        # tl：矩形定点
        # br：矩形的宽高
        # (0,0,255)：矩形边框颜色
        # 1：矩形边框大小
        cv2.rectangle(target, tl, br, (0, 0, 255), 2)
        # cv2.imwrite(self.out, target)
        return tl[0]

    def clear_white(self, img):
        # 清除图片的空白区域，这里主要清除滑块的空白
        rows, cols, channel = img.shape
        min_x = 255
        min_y = 255
        max_x = 0
        max_y = 0
        for x in range(1, rows):
            for y in range(1, cols):
                t = set(img[x, y])
                if len(t) >= 2:
                    if x <= min_x:
                        min_x = x
                    elif x >= max_x:
                        max_x = x

                    if y <= min_y:
                        min_y = y
                    elif y >= max_y:
                        max_y = y
        img1 = img[min_x:max_x, min_y: max_y]
        return img1


class AES(object):

    def encrypt(self, aes_key: str, plaintext: str) -> str:
        """ 对明文进行加密 """
        cipher = _AES.new(
            key=bytes(aes_key, encoding='utf-8'),
            mode=_AES.MODE_CBC,
            iv=bytes(aes_key, encoding='utf-8')
        )
        result = base64.b64encode(cipher.encrypt(pad(plaintext.encode('utf-8'), 16))).decode('utf-8')
        result = quote_plus(result)
        return result

    def decrypt(self, aes_key: str, ciphertext: str) -> str:
        """ 对密文进行解密 """
        cipher = _AES.new(
            key=bytes(aes_key, encoding='utf-8'),
            mode=_AES.MODE_CBC,
            iv=bytes(aes_key, encoding='utf-8')
        )
        result = unpad(cipher.decrypt(base64.b64decode(ciphertext)), 16).decode('utf-8')
        return result


class AnjukeSlide(object):

    def __init__(self):
        self.sess = Session()
        self.aes = AES()
        self.gap = GapGeneral((280, 158), (56, 158))

    def get_session_id(self) -> str:
        """ 获取 sessionId """
        url = 'https://www.anjuke.com/captcha-verify/?callback=shield&from=antispam'
        response = self.sess.get(url)
        session_id = re.search('<input type="hidden" name="sessionId" id="sessionId" value="(.*?)" />', response.text).group(1)
        logger.info(f'成功获取sessionId ==> {session_id}')
        return session_id

    def calc_key_iv(self, session_id: str) -> str:
        """ 根据 sessionId 计算 aesKey aesIv """
        c = ''
        for idx, ele in enumerate(session_id):
            if idx % 2 != 0:
                c += ele
        logger.info(f'处理sessionId获取key ==> {c}')
        return c

    def getInfoTp(self, session_id: str, d_info: str) -> dict:
        """ 请求第一个接口 https://verifycode.58.com/captcha/getInfoTp 拿到验证码图片信息 """
        url = 'https://verifycode.58.com/captcha/getInfoTp'
        data = {
            'sessionId': session_id,
            'showType': 'embed',
            'track': '',
            'clientType': '1',
            'clientId': '1',
            'language': 'zh-CN',
            'dInfo': d_info,
        }
        resp = self.sess.post(url, data)
        logger.info(f'getInfoTp接口返回: {resp.json()}')
        return resp.json()

    def checkInfoTp(self, session_id: str, response_id: str, d_info: str, data: str):
        """ 请求验证接口 https://verifycode.58.com/captcha/checkInfoTp """
        url = 'https://verifycode.58.com/captcha/checkInfoTp'
        data = {
            'sessionId': session_id,
            'responseId': response_id,
            'dInfo': d_info,
            'language': 'zh-CN',
            'data': data,
        }
        resp = self.sess.post(url, data)
        logger.info(f'checkInfoTp接口返回: {resp.json()}')
        return resp.json()

    def generate_track(self, distance) -> str:
        # 生成轨迹
        all_track = ['26,12,1', '26,13,78', '27,13,166', '28,13,174', '29,13,191', '30,13,207', '31,13,223', '32,13,230', '33,13,246', '34,13,262', '35,13,279', '36,13,286', '37,13,303', '38,13,327', '39,13,334', '40,13,342', '41,13,374', '42,13,391', '43,13,414', '44,13,422', '45,13,438', '46,13,454', '48,13,470', '49,13,495', '50,13,519', '51,13,535', '52,13,542', '53,13,567', '54,13,583', '55,13,619', '56,13,622', '57,13,638', '58,13,663', '59,13,679', '60,13,694', '61,13,711', '62,13,727',
                     '63,13,735', '63,14,742', '64,15,750', '65,15,774', '66,15,798', '67,15,807', '68,15,822', '69,15,846', '70,15,862', '71,15,870', '72,15,894', '73,15,910', '74,15,926', '75,15,951', '76,15,958', '77,15,974', '79,15,998', '80,15,1023', '81,15,1054', '82,15,1086', '83,15,1094', '84,15,1126', '85,16,1134', '86,16,1166', '87,16,1175', '88,17,1190', '89,17,1214', '90,17,1222', '91,17,1238', '92,17,1263', '94,18,1279', '95,18,1318', '96,18,1336', '97,19,1342', '98,19,1358',
                     '99,19,1374', '100,19,1390', '101,19,1399', '102,19,1414', '103,19,1446', '104,19,1462', '105,19,1470', '106,19,1487', '107,19,1510', '108,19,1518', '109,19,1526', '110,19,1542', '111,19,1558', '112,19,1574', '113,19,1590', '114,19,1607', '115,20,1630', '116,20,1654', '117,20,1662', '118,20,1679', '119,20,1687', '120,20,1694', '121,20,1703', '122,20,1726', '123,20,1734', '124,20,1758', '125,20,1774', '126,20,1790', '127,20,1814', '128,20,1822', '129,20,1838', '131,20,1854',
                     '132,20,1870', '133,20,1886', '134,20,1894', '135,20,1902', '136,20,1934', '137,20,1982', '138,20,2006', '139,20,2015', '140,20,2030', '141,20,2047', '142,20,2054', '143,20,2062', '144,20,2078', '145,20,2086', '146,20,2102', '147,20,2126', '148,20,2134', '149,20,2158', '150,19,2174', '151,19,2198', '152,19,2215', '153,19,2222', '154,19,2239', '157,18,2254', '158,18,2278', '160,17,2296', '161,17,2310', '163,17,2326', '164,17,2342', '166,17,2350', '167,17,2358', '168,17,2367',
                     '169,17,2374', '171,17,2383', '171,16,2390', '173,15,2422', '174,15,2471', '175,15,2486', '176,15,2502', '177,15,2510', '178,15,2518', '178,14,2526', '179,14,2534', '180,14,2550', '181,14,2574', '182,14,2590', '183,14,2598', '184,14,2614', '185,14,2622', '186,14,2630', '187,14,2639', '188,14,2646', '190,14,2662', '191,14,2678', '192,14,2686', '193,14,2694', '194,14,2710', '195,14,2734', '197,14,2750', '198,14,2766', '199,14,2776', '200,14,2782', '201,14,2790', '202,14,2798',
                     '204,14,2814', '205,14,2838', '206,14,2854', '207,14,2878', '208,14,2910', '209,14,2934', '211,14,2951', '212,14,2966', '213,14,2990', '214,14,3014', '215,14,3022', '216,14,3055', '217,14,3078', '218,14,3110', '219,14,3120', '220,14,3142', '221,14,3174', '222,14,3191', '223,14,3199', '224,14,3214', '225,14,3238', '226,14,3263', '227,14,3271', '228,14,3280', '229,14,3294', '230,14,3319', '231,14,3334', '232,14,3358', '233,14,3422', '234,14,3446', '235,14,3470', '236,14,3486',
                     '237,14,3518', '238,14,3567', '239,14,3574', '240,14,3638', '241,14,3662', '242,14,3702', '243,14,3718', '244,14,3766', '245,14,3807', '246,14,3830', '247,14,3854', '247,14,4151']
        now_track = []
        for track in all_track:
            if distance + 26 < int(track.split(',')[0]):
                break
            else:
                now_track.append(track)
        return now_track

    def main(self):
        content = '{"sdkv":"3.0.1","busurl":"https://www.anjuke.com/captcha-verify/?callback=shield&from=antispam","useragent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36","clienttype":"1"}'
        session_id = self.get_session_id()
        aes_key_iv = self.calc_key_iv(session_id)
        d_info = self.aes.encrypt(aes_key_iv, content)
        logger.info(f'dInfo ==> {d_info}')
        get_info_tp_data = self.getInfoTp(session_id, d_info)
        get_info_tp_data_plain = json.loads(self.aes.decrypt(aes_key_iv, get_info_tp_data['data']['info']))
        logger.info(f'getInfoTp接口返回解密为: {get_info_tp_data_plain}')
        distance = self.gap.get_distance(get_info_tp_data_plain['bgImgUrl'], get_info_tp_data_plain['puzzleImgUrl'])
        logger.info(f'缺口位置: {distance}')
        track_data = self.generate_track(distance)
        check_data_plaintext = {'x': distance, 'track': track_data, 'p': [0, 0]}
        check_data_ciphertext = self.aes.encrypt(aes_key_iv, json.dumps(check_data_plaintext).replace(' ', ''))
        logger.info(f'checkInfoTp data: {check_data_ciphertext}')
        check_result = self.checkInfoTp(session_id, get_info_tp_data['data']['responseId'], d_info, check_data_ciphertext)


if __name__ == '__main__':
    anjuke_slide = AnjukeSlide()
    anjuke_slide.main()
