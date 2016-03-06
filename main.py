import random
import math
import numpy as np
import cv2
import array
import sys

k_means = 10

# number of features in each set
num_features = 64
num_runs = 5




def LoadData(filename):
    # """
    # Each line in the datafile is a csv with features values, followed by a single label (0 or 1),
    # per sample; one sample per line
    # """

    # "The file function reads the filename from the current directory, unless you provide an absolute path
    # e.g. /path/to/file/file.py or C:\\path\\to\\file.py"

    unprocessed_data_file = file(filename,'r')

    "Obtain all lines in the file as a list of strings."

    unprocessed_data = unprocessed_data_file.readlines()

    labels = []
    features = []

    for line in unprocessed_data:
        feature_vector = []

        "Convert the String into a list of strings, being the elements of the string separated by commas"
        split_line = line.split(',')

        "Iterate across elements in the split_line except for the final element "
        for element in split_line[:-1]:
            feature_vector.append(float(element))

        "Add the new vector of feature values for the sample to the features list"
        features.append(feature_vector)

        "Obtain the label for the sample and add it to the labels list"
        labels.append(int(split_line[-1]))

    "Return List of features with its list of corresponding labels"
    return features, labels


def init_centers():
    the_center_list = []
    for i in range(k_means):
        one_center = []
        for j in range(num_features):
            one_center.append(random.randint(0, 16))
        the_center_list.append(one_center)

    return the_center_list


def get_distance_squared(one_center, one_set):
    sum_all = 0
    for i in range(num_features):
        sum_all += math.pow((one_set[i] - one_center[i]), 2)
    return sum_all


def wrapper_get_distance_squared(the_train, the_centers):
    the_distances = []
    for i in range(k_means):
        one_center_distances = []
        for j in range(len(the_train)):
            one_center_distances.append(get_distance_squared(the_centers[i], the_train[j]))
        the_distances.append(one_center_distances)
    return the_distances


def find_min_index(the_list):
    temp = min(the_list)
    index = -1
    for i in range(len(the_list)):
        if temp == the_list[i]:
            index = i
    return index


def wrapper_find_min_index(the_distances):
    the_clustered_sets = []
    for i in range(k_means):
        the_clustered_sets.append([])

    for i in range(len(the_distances[0])):
        temp = []
        for j in range(k_means):
            temp.append(the_distances[j][i])

        the_min_index = find_min_index(temp)
        the_clustered_sets[the_min_index].append(i)
    return the_clustered_sets


def build_set_for_each_center(one_clustered_set, the_train_featrues):
    temp = []
    for i in range(len(one_clustered_set)):
        temp.append(the_train_featrues[one_clustered_set[i]])
    return temp


def update_centers(the_clustered_sets, the_train_features):
    temp = []
    for i in range(k_means):
        sets_in_one_cluster = build_set_for_each_center(the_clustered_sets[i], the_train_features)
        temp.append(np.mean(np.asarray(sets_in_one_cluster), axis=0).tolist())
    return temp


def stop_condition(the_pre_centers, the_cur_centers):
    delta_sum_distance = 0
    for i in range(k_means):
        # print "distance: ", math.sqrt(get_distance_squared(the_pre_centers[i], the_cur_centers[i]))
        if math.sqrt(get_distance_squared(the_pre_centers[i], the_cur_centers[i])) > 0.001:
            return False
    return True


def copy_centers(the_list):
    dest = []
    for i in range(k_means):
        temp = []
        for j in range(num_features):
            temp.append(the_list[i][j])
        dest.append(temp)
    return dest


def check_empty(the_clustered_sets):
    for i in range(k_means):
        if len(the_clustered_sets[i]) == 0:
            return True
    return False


def get_sum_squared_error(the_clustered_sets, the_cluster_centers, the_train_features):
    sum_all = 0
    for i in range(k_means):
        for j in range(len(the_clustered_sets[i])):
            index = the_clustered_sets[i][j]
            sum_all += get_distance_squared(the_cluster_centers[i], the_train_features[index])
    return sum_all


def get_sum_squared_separation(the_cluster_centers):
    sum_all = 0
    for i in range(k_means-1):
        j = i + 1
        while j < k_means:
            sum_all += get_distance_squared(the_cluster_centers[i], the_cluster_centers[j])
            j += 1

    return sum_all


def get_entropy(one_clustered_set, the_train_labels):
    counter =[]
    for i in range(k_means):
        counter.append(0)

    for i in range(len(one_clustered_set)):
        index = one_clustered_set[i]
        counter[the_train_labels[index]] += 1

    total = len(one_clustered_set)
    sum_entropy = 0
    for i in range(k_means):
        if counter[i] == 0:
            sum_entropy += 0
        else:
            sum_entropy += (counter[i]/float(total)) * math.log((counter[i]/float(total)), 2)

    return -sum_entropy


def get_mean_entropy(the_clustered_sets, the_train_labels):
    sum_all_entropy = 0
    total_train_sets = len(the_train_labels)
    for i in range(k_means):
        one_set_length = len(the_clustered_sets[i])

        sum_all_entropy += (one_set_length/float(total_train_sets)) * get_entropy(the_clustered_sets[i], the_train_labels)

    return sum_all_entropy


def train(the_train_features, the_train_labels):
    count_while = 0
    count_for = 0
    sum_squared_error_list = []
    sum_squared_separation_list = []
    mean_entropy_list = []
    cluster_centers_list = []
    min_sse_index = -1
    wasted_count = 0

    while count_for < num_runs:
        cluster_centers = init_centers()
        distances = wrapper_get_distance_squared(the_train_features, cluster_centers)
        clustered_sets = wrapper_find_min_index(distances)
        center_bool = True
        if check_empty(clustered_sets) == False:
            while center_bool == True:
                pre_centers = copy_centers(cluster_centers)
                cluster_centers = update_centers(clustered_sets, the_train_features)
                distances = wrapper_get_distance_squared(the_train_features, cluster_centers)
                clustered_sets = wrapper_find_min_index(distances)
                if stop_condition(pre_centers, cluster_centers) == True:
                    center_bool = False
                count_while += 1
                # print "while: ", count_while
            count_for += 1
            print "for: ", count_for

            sum_squared_error = get_sum_squared_error(clustered_sets, cluster_centers, the_train_features)
            sum_squared_separation = get_sum_squared_separation(cluster_centers)
            mean_entropy = get_mean_entropy(clustered_sets, the_train_labels)

            sum_squared_error_list.append(sum_squared_error)
            sum_squared_separation_list.append(sum_squared_separation)
            mean_entropy_list.append(mean_entropy)
            temp_centers = copy_centers(cluster_centers)
            cluster_centers_list.append(temp_centers)

            min_sse_index = find_min_index(sum_squared_error_list)
            # print sum_squared_error_list[min_sse_index]
            # print sum_squared_separation_list[min_sse_index]
            # print mean_entropy_list[min_sse_index]

        # wasted_count += 1
        # print "wasted_count: ", wasted_count
    return cluster_centers_list[min_sse_index], sum_squared_error_list[min_sse_index], \
           sum_squared_separation_list[min_sse_index], mean_entropy_list[min_sse_index]


def find_most_freq_class(one_test_clustered_sets, the_test_labels):
    counter = []
    for i in range(k_means):
        counter.append(0)

    for i in range(len(one_test_clustered_sets)):
        index = one_test_clustered_sets[i]
        counter[the_test_labels[index]] += 1

    temp = []
    max_class = max(counter)
    for i in range(k_means):
        if counter[i] == max_class:
            temp.append(i)
    random.shuffle(temp)
    return temp[0]


def get_clustered_sets_label(the_test_clustered_sets, the_test_labels):
    temp_clustered_sets_label = []

    for i in range(k_means):
        index = find_most_freq_class(the_test_clustered_sets[i], the_test_labels)
        temp_clustered_sets_label.append(index)
    return temp_clustered_sets_label


def get_confusion_matrix(the_test_clustered_sets, the_test_clustered_sets_label, the_test_label):
    temp_confusion_matrix = []
    for i in range(k_means):
        temp = []
        for j in range(k_means):
            temp.append(0)
        temp_confusion_matrix.append(temp)

    for i in range(len(the_test_clustered_sets)):
        for j in range(len(the_test_clustered_sets[i])):
            index = the_test_clustered_sets[i][j]
            actual_index = the_test_label[index]
            predict_index = the_test_clustered_sets_label[i]
            temp_confusion_matrix[predict_index][actual_index] += 1

    return temp_confusion_matrix


def get_acc(the_confusion_matrix):
    sum_all = 0
    diagonal = 0
    for i in range(len(the_confusion_matrix)):
        for j in range(len(the_confusion_matrix[i])):
            sum_all += the_confusion_matrix[i][j]
            if i == j:
                diagonal += the_confusion_matrix[i][j]

    return diagonal/float(sum_all)


def print_as_pgm(one_cluster_center, the_index):
    # print "one center: "

    # define the width  (columns) and height (rows) of your image
    width = 8
    height = 8

    # declare 1-d array of unsigned char and assign it random values
    buff=array.array('B')

    for i in range(0, width*height):
      buff.append(int(round(one_cluster_center[i])) * 16)


    # open file for writing
    filename = str(the_index) + ".pgm"

    try:
      fout=open(filename, 'wb')
    except IOError, er:
      print "Cannot open file "
      sys.exit()


    # define PGM Header
    pgmHeader = 'P5' + '\n' + str(width) + '  ' + str(height) + '  ' + str(255) + '\n'

    # write the header to the file
    fout.write(pgmHeader)

    # write the data to the file
    buff.tofile(fout)

    # close the file
    fout.close()



def wrapper_print_as_pgm(the_cluster_centers):
    for i in range(len(the_cluster_centers)):
        print_as_pgm(the_cluster_centers[i], i)



train_features, train_labels = LoadData("optdigits.train")
test_features, test_labels = LoadData("optdigits.test")

final_cluster_centers, final_sum_squared_error, final_sum_squared_separation,\
    final_mean_entropy = train(train_features, train_labels)


test_distances = wrapper_get_distance_squared(test_features, final_cluster_centers)
test_clustered_sets = wrapper_find_min_index(test_distances)

# for element in test_clustered_sets:
#     print "one cluster: ", len(element)

test_clustered_sets_label = get_clustered_sets_label(test_clustered_sets, test_labels)

# print "mapped label: ", test_clustered_sets_label

confusion_matrix = get_confusion_matrix(test_clustered_sets, test_clustered_sets_label, test_labels)

# for element in range(k_means):
#     print confusion_matrix[element]
acc = get_acc(confusion_matrix)


print "accuracy: ", acc
print "final_sum_squared_error: ", final_sum_squared_error
print "final_sum_squared_separation: ", final_sum_squared_separation
print "final_mean_entropy: ", final_mean_entropy

print "confusion matrix: "
print np.asarray(confusion_matrix)

wrapper_print_as_pgm(final_cluster_centers)
print test_clustered_sets_label

# print len(train)
# print len(train[0])
# print len(test)
# print len(cluster_centers)
# print len(cluster_centers[0])
#
# print len(distances)
# print len(distances[0])
#
# print len(clustered_sets)
