import numpy as np

from Auxiliary import softmax, net_lay
from .CRUD_files import read_scales_to_matrix


def neuron_net(one_layer, scales_index):
    # layer_matrices = pd.DataFrame()
    layer_arr = np.empty((scales_index + 2,), dtype=object)
    layer_arr[0] = np.array(one_layer.to_numpy().reshape(-1, 1), dtype=float, )
    for j in range(1, scales_index + 1):

        file_path = f'scales_{j}.csv'
        scales_matrix = read_scales_to_matrix(file_path)

        if scales_matrix is not None:
            one_layer = net_lay(one_layer, scales_matrix)

            layer_arr[j] = np.array(one_layer.to_numpy().reshape(-1, 1), dtype=float, )
            # new_row = pd.DataFrame(one_layer)
            # layer_matrices = pd.concat([layer_matrices, new_row.T], ignore_index=True)
        else:
            print("Error: No file with neuron weights")

    scales_matrix = read_scales_to_matrix('scales_end.csv')
    one_layer = net_lay(one_layer, scales_matrix, True)
    layer_arr[scales_index + 1] = np.array(one_layer.to_numpy().reshape(-1, 1), dtype=float, )
    result = softmax(one_layer)

    return result, layer_arr
