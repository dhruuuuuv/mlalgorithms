import csv
import math
import operator


def crossval(k, dataset):
    number_items = len(dataset)
    length_fold = math.floor(len(dataset) / k)
    # produce_sets(1, dataset, length_fold)
    # print(length_fold)
    crossval_sets = []
    for x in range(1, k+1):
        crossval_sets.append(produce_sets(x, k, dataset, length_fold))
    return crossval_sets


def produce_sets(i, k, dataset, length_fold):
    endval = i*length_fold
    startval = endval - length_fold
    test = dataset[startval:endval]
    training = []
    reststartval = 0
    restendval = len(dataset)
    if i == 0:
        reststartval = endval
        training = dataset[reststartval:restendval]
    if i == k:
        restendval = startval
        training = dataset[reststartval:restendval]
    else:
        training += dataset[0:startval]
        training += dataset[endval:len(dataset)]
    return (training, test)

# def
# crossval(5)

# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# produce_sets(3, 10, data, 2)
# print(crossval(5, data))
