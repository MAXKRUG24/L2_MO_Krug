import os

import numpy as np
from dotenv import load_dotenv

from Auxiliary import softmax
from .CRUD_files import read_img_to_matrix
from .NeuronNet import neuron_net
from .Metrics import adding_metrics, binary_cross_entropy


def validation(epochs=1):
    img_matrix = read_img_to_matrix('./Files/test.csv')

    load_dotenv()
    scales_index = int(os.getenv('scales_index'))
    count_class = int(os.getenv('count_class'))

    for epoch in range(epochs):
        if img_matrix is not None:
            number_err = []
            data_for_metrics = np.zeros((count_class, 4))


            row = img_matrix[0].shape[0]

            for i in range(row):

                layer_matrices = img_matrix[0].iloc[i]
                end_y, layer_matrices = neuron_net(layer_matrices, scales_index)
                # print(f"Result {i} for class {img_matrix[1][i][0] - 1}:\n{end_y}")

                true_answer = img_matrix[1][i][0] - 1

                get_answer = end_y.idxmax()
                for j in range(count_class):
                    if get_answer == j and true_answer == j:
                        data_for_metrics[j][0] += 1
                    elif get_answer != j and true_answer == j:
                        data_for_metrics[j][1] += 1
                    elif get_answer == j and true_answer != j:
                        data_for_metrics[j][2] += 1
                    elif get_answer != j and true_answer != j:
                        data_for_metrics[j][3] += 1

                answer = np.zeros(10)
                answer[true_answer] = 1
                loss = binary_cross_entropy(answer, softmax(layer_matrices[3]))
                number_err.append(loss)

            adding_metrics("validate_metrics", count_class, data_for_metrics, number_err)
        print(f"Later {epoch + 1} epochs.")
