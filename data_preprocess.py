# -*- coding:utf-8 -*-
# author:zhangwei

def data_pre(filename):
    with open(filename , 'r') as fr:
        with open('/home/zhangwei/data/AgriculturalDisease_trainingset/agri_data_01.txt' , 'w') as fw:
            lines = fr.readlines()
            i = 1
            for line in lines:
                line = line.strip()
                # print(line)
                if i % 2 == 0:
                    fw.write(line + '\n')
                else:
                    fw.write(line + '?')
                i += 1

if __name__ == '__main__':
    pathname = '/home/zhangwei/data/AgriculturalDisease_trainingset/agri_data.txt'
    data_pre(filename=pathname)