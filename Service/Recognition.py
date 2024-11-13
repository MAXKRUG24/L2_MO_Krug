import os

from dotenv import load_dotenv

from . import read_one_img_to_matrix, neuron_net


def recognition(img_path):
    img_matrix = read_one_img_to_matrix(img_path)

    load_dotenv()
    scales_index = int(os.getenv('scales_index'))
    layer_matrices = img_matrix.iloc[0]

    recognizer = neuron_net(layer_matrices, scales_index)[0]
    get_answer = recognizer.idxmax()
    services = {
        0: "1",
        1: "2",
        2: "3",
        3: "4",
        4: "5",
        5: "6",
        6: "7",
        7: "8",
        8: "9",
        9: "10"
    }

    return services.get(get_answer, "Invalid value. Please enter a number from 1 to 10.")
