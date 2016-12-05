import csv
import math
import operator

import matplotlib.pyplot as plt
import numpy as np

import KNN
import crossval
import datanorm
import linearreg

# fn to load in the dataset
def import_dataset(filename):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile, delimiter=' ')
        data = list(lines)
        for i in range(len(data)):
            point = list(map(float, data[i]))
            data[i] = point
        return data

def main():

    # import datasets
    training_org = import_dataset("IrisTrainML.dt")
    test_org = import_dataset("IrisTestML.dt")

    # 1.1
    print("Part 1.1\n")

    # run KNN for K = 1, 3, 5
    ks = [1, 3, 5]
    # run on training_org set
    print("on training data")
    for k in ks:
        loss = KNN.knearest(k, training_org, training_org)
        print("KNN algorithm with K = " + repr(k))
        print('loss: ' + repr(loss))

    # run on test_org set
    print("on test data")
    for k in ks:
        loss = KNN.knearest(k, training_org, test_org)
        print("KNN algorithm with K = " + repr(k))
        print('loss: ' + repr(loss))

    # ---
    # 1.2
    print("\nPart 1.2\n")
    ks = range(1, 26 , 2)
    avg_crossval = []

    # produce the cross validation training_org and test_org pairs
    sets = crossval.crossval(5, training_org)


    # for each value of k
    for k in ks:
        knn_results = []

        # for each 5 of the cross validation training, test sets
        for (training, test) in sets:
            # calculate the loss of the KNN algorithm
            knn_results.append(KNN.knearest(k, training, test))

        # caluclate the average of the 5 tests
        # print(knn_results)
        avg = 0
        for x in knn_results:
            avg += x
        avg = avg / len(knn_results)

        # add the average as a tuple with the k value to the final set
        avg_crossval.append((k, avg))

    # print(avg_crossval)

    # plot graph

    # x_val = [x[0] for x in avg_crossval]
    # y_val = [x[1] for x in avg_crossval]
    #
    # plt.plot(x_val, y_val)
    # plt.title('Cross-validation loss of K-Nearest Neighbour\'s Algorithm against values of K')
    # plt.xlabel('K values')
    # plt.ylabel('Average 5 Fold Cross-Validation Loss of KNN algorithm')
    # plt.xlim((0, max(x_val)))
    # plt.ylim((min(y_val)-0.25, 5.25))
    # plt.grid(True)
    # plt.show()

    # calculate the minimum K value from the data
    currentmink = 0
    currentminloss = 100
    for (k,loss) in avg_crossval:
        if loss <= currentminloss:
            currentminloss = loss
            currentmink = k

    # run on training_org and test_org set
    k = currentmink
    print("on training data")
    loss = KNN.knearest(k, training_org, training_org)
    print("KNN algorithm with K = " + repr(k))
    print('loss: ' + repr(loss))

    # run on test_org set
    print("on test data")
    loss = KNN.knearest(k, training_org, test_org)
    print("KNN algorithm with K = " + repr(k))
    print('loss: ' + repr(loss))

    # ---
    # 1.3
    print("\nPart 1.3\n")
    train_mean = datanorm.calc_mean(training_org)
    train_var = datanorm.calc_var(training_org)

    print("\nfor training data")
    print("mean equals " + repr(train_mean))
    print("var equals " + repr(train_var))

    # normalise the test and training data
    normalised_train = datanorm.normalise(training_org)
    normalised_test = datanorm.normalise(test_org)
    # re-add the classification to the normalised data
    normalised_test = datanorm.readd_classification(test_org, normalised_test)
    normalised_train = datanorm.readd_classification(training_org, normalised_train)

    # calculate the mean and variance of the normalised test data
    test_mean = datanorm.calc_mean(normalised_test)
    test_var = datanorm.calc_var(normalised_test)

    print("\nfor transformed test data")
    print("mean equals " + repr(test_mean))
    print("var equals " + repr(test_var))

    # print(training_org)
    # print(datanorm.calc_var(normalised_train))

    ks = range(1, 26 , 2)
    avg_crossvalx = []

    # produce the cross validation training_org and test_org pairs
    sets = crossval.crossval(5, normalised_train)


    # for each value of k
    for k in ks:
        knn_results = []

        # for each 5 of the cross validation training, test sets
        for (trainingx, testx) in sets:
            # print(training, test)
            # calculate the loss of the KNN algorithm
            knn_results.append(KNN.knearest(k, trainingx, testx))

        # caluclate the average of the 5 tests
        # print(knn_results)
        avg = 0
        for x in knn_results:
            avg += x
        avg = avg / len(knn_results)

        # add the average as a tuple with the k value to the final set
        avg_crossvalx.append((k, avg))


    # plot graph
    x_val = [x[0] for x in avg_crossvalx]
    y_val = [x[1] for x in avg_crossvalx]

    # calculate the minimum K value from the data
    currentmink = 0
    currentminloss = 100
    for (k,loss) in avg_crossvalx:
        if loss <= currentminloss:
            currentminloss = loss
            currentmink = k

    # run on training_org and test_org set
    k = currentmink
    print("\non training data")
    loss = KNN.knearest(k, normalised_train, normalised_train)
    print("KNN algorithm with K = " + repr(k))
    print('loss: ' + repr(loss))

    # run on test_org set
    print("\non test data")
    loss = KNN.knearest(k, normalised_train, normalised_test)
    print("KNN algorithm with K = " + repr(k))
    print('loss: ' + repr(loss))

    # plt.plot(x_val, y_val)
    # plt.title('Cross-validation loss of K-Nearest Neighbour\'s Algorithm against values of K')
    # plt.xlabel('K values')
    # plt.ylabel('Average 5 Fold Cross-Validation Loss of KNN algorithm')
    # plt.xlim((0, max(x_val)))
    # plt.ylim((min(y_val)-0.25, 5.25))
    # plt.grid(True)
    # plt.show()

    # ---
    # 5.1
    print("\nPart 5.1\n")

    (w, b) = linearreg.linear_regression("DanWood.dt")


    danwood = import_dataset("DanWood.dt")
    x_valdan = [x[0] for x in danwood]
    y_valdan = [x[1] for x in danwood]


    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot the original points
    ax.scatter(x_valdan, y_valdan)
    plt.title('Plot of DanWood input set and linear regression line')
    plt.xlabel('Absolute Temperature (Units of 1000 degrees Kelvin)')
    plt.ylabel('Energy Raditation per cm^2 per sec')
    plt.grid(True)

    # plot the straight line given by the equation wx + b
    ylinreg = [ (w*x + b) for x in x_valdan]
    ax.plot(x_valdan, ylinreg)

    print("w = " + repr(w))
    print("b = " + repr(b))

    # calculate the mean squared error
    mse = linearreg.mse(x_valdan, ylinreg)

    print("mean squared error of the prediction is: " + repr(mse))

    plt.show()

if __name__ == '__main__':
    main()
