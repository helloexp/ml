# -*- coding:utf-8 -*-i
import os
import random
import numpy as np


from PIL import Image
def get_imgs(rate=0.2):

    path="/Users/tong/Downloads/ml-exe/Discuz"
    imgs = os.listdir(path)
    random.shuffle(imgs)

    print(type(imgs))

    img_num = len(imgs)

    test_num= int((img_num * rate) / (1+rate))

    test_imgs=imgs[:test_num]
    # 根据文件名获取测试集标签
    test_labels = list(map(lambda x: x.split('.')[0], test_imgs))
    # 训练集
    train_imgs = imgs[test_num:]
    # 根据文件名获取训练集标签
    train_labels = list(map(lambda x: x.split('.')[0], train_imgs))

    return test_imgs, test_labels, train_imgs, train_labels




def text2vec(text):
    """
    文本转向量
    Parameters:
      text:文本
    Returns:
      vector:向量
    """
    if len(text) > 4:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(4 * 63)

    for i, c in enumerate(text):
        idx = i * 63 + char2pos(c)
        vector[idx] = 1
    return vector

def char2pos(c):
    if c == '_':
        k = 62
        return k
    k = ord(c) - 48
    if k > 9:
        k = ord(c) - 55
        if k > 35:
            k = ord(c) - 61
            if k > 61:
                raise ValueError('No Map')
    return k


def vec2text(vec):
    """
    向量转文本
    Parameters:
      vec:向量
    Returns:
      文本
    """
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_at_pos = i  # c/63
        char_idx = c % 63
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)




if __name__ == '__main__':
    test_imgs, test_labels, train_imgs, train_labels=get_imgs()



