import csv
import math
import operator

# calculate the mean of a dataset, assuming the last row is for feature id
def calc_mean(dataset):
    mean_features = map(mean, zip(*dataset))
    mean_features = list(mean_features)
    # if remove_final == 1:
    del mean_features[-1]
    return mean_features

# calculate the var of a dataset, assuming the last row is for feature id
def calc_var(dataset):
    mf = calc_mean(dataset)
    compressed_set = list(zip(*dataset))
    # if remove_final == 1:
    del compressed_set[-1]
    recalc_set = []

    for i in range(len(compressed_set)):
        f = lambda x: (x - mf[i])**2
        adj_feature_set = list(map(f, compressed_set[i]))
        recalc_set.append(adj_feature_set)

    var_features = list(map(var, recalc_set))
    # print(var_features)
    return(var_features)

# calculate the sd
def calc_sd(dataset):
    vf = calc_var(dataset)
    sdf = list(map(lambda x : x ** 0.5, vf))
    return sdf

# mean fn to use in map
def mean(a):
    return (sum(a) / len(a))

def var(a):
    return (sum(a) / (len(a) - 1))

# return (mean, variance) as a tuple for each feature
def mean_variance(dataset):
    mf = calc_mean(dataset)
    vf = calc_var(dataset)
    return list(zip(mf, vf))

# normalise the data by ((x - mu) / sigma) so mu = 0, var = 1
def normalise(dataset):
    mf = calc_mean(dataset)
    sdf = calc_sd(dataset)

    compressed_set = list(zip(*dataset))
    del compressed_set[-1]
    recalc_set = []

    for i in range(len(compressed_set)):
        adj_feature_set = list(map(lambda x: ((x - mf[i]) / sdf[i]), compressed_set[i]))
        recalc_set.append(adj_feature_set)

    # print(dataset)
    # print(calc_mean(dataset, 1))
    # print(calc_sd(dataset, 1))
    # print(list(zip(*recalc_set)))

    # var_features = list(map(mean, recalc_set))
    # print(var_features)
    return list(zip(*recalc_set))

def readd_classification(original, normalised):
        for i in range(len(normalised)):
            rationalised_data = []
            entry = normalised[i]
            rationalised_data += entry
            rationalised_data += [original[i][-1]]
            normalised[i] = rationalised_data
        return normalised




# ds = [[1, 2, 3], [5, 3, 2], [19, 2, 3], [9, 10, 11]]
# print(calc_mean(ds))

# var(ds, 3)
# n = normalise(ds)
# print(calc_mean(n, 0))
# print(calc_var(n, 0))
