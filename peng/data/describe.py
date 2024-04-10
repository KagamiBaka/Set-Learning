import json 
import numpy
dirs = ['14lap', '14res', '15res', '16res']
files = ['/train_convert.json', '/dev_convert.json', '/test_convert.json']
for dir in dirs:
    for file in files:
        file = open('../../data/penga/' + dir + file)
        datas = json.load(file)
        max_tuple_num = 0
        count = numpy.zeros(6)
        for i in range(0, len(datas)):
            count[len(datas[i]['aspects']) - 1] += 1
            if len(datas[i]['aspects']) > max_tuple_num:
                max_tuple_num = len(datas[i]['aspects'])
        print(count)
        ratio = numpy.zeros(6)
        for i in range(len(count)):
            ratio[i] = count[i] / sum(count)
        print(ratio)
        print(max_tuple_num)
        