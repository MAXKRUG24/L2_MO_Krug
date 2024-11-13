import os
import pandas as pd
import csv
import random
from dotenv import load_dotenv

load_dotenv()
path_scales = os.getenv('path_scales')


def read_one_img_to_matrix(file_path, delimiter=';'):
    with open(file_path, 'r') as file:
        # Read lines, split by delimiter, and convert to float
        lines = file.readlines()
        arr = []
        for line in lines:
            # Split line into values and ignore comments
            values = line.split(delimiter)

            arr.append([float(value) for value in values])

        matrix = pd.DataFrame(arr)
        return matrix

def read_img_to_matrix(file_path, delimiter=';'):
    with open(file_path, 'r') as file:
        # Read lines, split by delimiter, and convert to float
        lines = file.readlines()
        last_two_values = []
        arr = []
        for line in lines:
            # Split line into values and ignore comments
            values = line.split(delimiter)

            arr.append([float(value) for value in values[:-2]])
            last_two_values.append([int(values[-2]), int(values[-1])])

        matrix = pd.DataFrame(arr)
        return matrix, last_two_values


def read_scales_to_matrix(file_name, delimiter=';') -> pd.DataFrame:
    with open(path_scales + file_name, 'r') as file:
        lines = file.readlines()
        arr = []
        for line in lines:
            values = line.split(delimiter)
            arr.append([float(value) for value in values])
        matrix = pd.DataFrame(arr)
        return matrix


def create_scales_file(file_name, num_rows, num_columns):
    # Создаем и открываем файл для записи
    with open(path_scales + file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        for _ in range(num_rows):
            row = [f"{random.uniform(-1, 1):.4f}" for _ in range(num_columns)]
            writer.writerow([';'.join(row)])
    return True


def write_scales_to_file(scale, file_name):
    if isinstance(scale, pd.DataFrame):
        scale = scale.iloc[1:]
        scale.to_csv(path_scales + file_name, index=False, sep=';')
    else:
        raise ValueError("scale должен быть экземпляром pandas DataFrame")


def write_arr_to_file(arr, file_name):
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(arr)
    return True
