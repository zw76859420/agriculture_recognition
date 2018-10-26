# -*- coding:utf-8 -*-
# author:zhangwei

import json


def read_path(filename):

    with open(filename , 'r') as fr:
        with open('/home/zhangwei/data/AgriculturalDisease_trainingset/agri_data.txt' , 'w') as fw:
            lines = json.loads(fr.read())
            for line in lines:
                # res = line.values()
                # print(res)
                # fw.write(str(res) + '\n')
                for v in line.values():
                    fw.write(str(v) + '\n')



if __name__ =='__main__':
    pathname = '/home/zhangwei/data/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json'
    read_path(pathname)