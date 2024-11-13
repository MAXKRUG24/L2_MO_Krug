import numpy as np

from .CRUD_files import write_arr_to_file


def binary_cross_entropy(t, p):
    t = t.reshape(t.shape[0], -1)
    result = t * np.log(p) + (1 - t) * np.log(1 - p)
    return -np.sum(result)


def adding_metrics(name_metrics, count_class, data_for_metrics, number_err):
    precision_arr = np.zeros(count_class)
    recall_arr = np.zeros(count_class)
    accuracy_arr = np.zeros(count_class)

    for i in range(count_class):
        if (data_for_metrics[i][0] + data_for_metrics[i][2]) != 0:
            precision_arr[i] += data_for_metrics[i][0] / (data_for_metrics[i][0] + data_for_metrics[i][2])
        if (data_for_metrics[i][0] + data_for_metrics[i][1]) != 0:
            recall_arr[i] += data_for_metrics[i][0] / (data_for_metrics[i][0] + data_for_metrics[i][1])
        if (data_for_metrics[i][0] + data_for_metrics[i][1] + data_for_metrics[i][2] + data_for_metrics[i][3]) != 0:
            accuracy_arr[i] += (data_for_metrics[i][0] + data_for_metrics[i][3]) / (
                    data_for_metrics[i][0] + data_for_metrics[i][1] + data_for_metrics[i][2] +
                    data_for_metrics[i][3])
    loss = np.mean(number_err)
    accuracy = np.mean(accuracy_arr)
    precision = np.mean(precision_arr)
    recall = np.mean(recall_arr)

    metric_arr = np.array([loss, accuracy, precision, recall])

    write_arr_to_file(metric_arr, f'./Files/metrics/{name_metrics}.csv')

    return metric_arr
