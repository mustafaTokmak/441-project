import math 
def scaled_manhattan_distance(train_data, test_data):
    arr = []
    feature_means = []
    mean_counter = 0
    for i in range(len(train_data[0])):
        feature_means.append(0)

    for train in train_data:
        mean_counter += 1
        for i in range(len(train)):
            feature_means[i] += train[i]

    for f in feature_means:
        f = f/mean_counter
    for test in test_data:
        sum_of_sum = 0
        counter = 0
        for train in train_data:
            counter += 1
            for i in range(len(test)):
                if(feature_means[i] == 0):
                    sum_of_sum += (math.fabs(test[i] - train[i]))
                else:
                    sum_of_sum += ((math.fabs(test[i] -
                                              train[i]))/(feature_means[i]))

        arr.append(-sum_of_sum/counter)
    # time.sleep(1)
    return arr
