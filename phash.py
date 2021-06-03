# encoding=utf-8
import numpy as np
import cv2
import os
def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])
# 计算两个哈希值之间的差异
def campHash(hash1, hash2):
    n = 0
    # hash长度不同返回-1,此时不能比较
    if len(hash1) != len(hash2):
        return -1
    # 如果hash长度相同遍历长度
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n
# 差异值哈希算法
def dhash(file):
    # 将图片转化为8*8
    img = cv2.imread(file, 0)
    gray = cv2.resize(img, (65, 64), interpolation=cv2.INTER_CUBIC)
    # 将图片转化为灰度图
    dhash_str = ''
    for i in range(64):
        for j in range(64):
            if gray[i, j] > gray[i, j + 1]:
                dhash_str = dhash_str + '1'
            else:
                dhash_str = dhash_str + '0'
    result = ''
    for i in range(0, 64*64, 4):
        result += ''.join('%x' % int(dhash_str[i: i + 4], 2))
    # print("dhash值",result)
    return result


def pHash(imgfile):
    """get image pHash value"""
    #加载并调整图片为32x32灰度图片
    img=cv2.imread(imgfile, 0)
    img=cv2.resize(img,(256,256),interpolation=cv2.INTER_CUBIC)

        #创建二维列表
    h, w = img.shape[:2]
    vis0 = np.zeros((h,w), np.float32)
    vis0[:h,:w] = img       #填充数据

    #二维Dct变换
    vis1 = cv2.dct(cv2.dct(vis0))
    #cv.SaveImage('a.jpg',cv.fromarray(vis0)) #保存图片
    vis1.resize(256,256)

    #把二维list变成一维list
    img_list=[n for a in vis1.tolist() for n in a ]

    #计算均值
    avg = sum(img_list)*1./len(img_list)
    avg_list = ['0' if i<avg else '1' for i in img_list]

    #得到哈希值
    return ''.join(['%x' % int(''.join(avg_list[x:x+4]),2) for x in range(0,256*256,4)])


if __name__ == '__main__':
    fh = open('val_result.txt', 'w', encoding='utf-8')
    count=0
    for parent, dirnames, filenames in os.walk('val'):
        for filename in filenames:
            d={}
            for parent1, dirnames1, filenames1 in os.walk('gallery'):
                for filename1 in filenames1:
                    Hash1=dhash('val/'+filename)
                    Hash2=dhash('gallery/'+filename1)
                    out_score = campHash(Hash2,Hash1)
                    d[filename1]=out_score
            a=sorted(d.items(), key=lambda item: item[1], reverse=False)
            fh.write(filename+" "+a[0][0].split('.')[0]+"\n")
            count=count+1
            print("第%d张图片"%count)
            print(a[0][0].split('.')[0])
    fh.close()
