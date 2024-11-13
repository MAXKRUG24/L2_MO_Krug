import csv
import random

from PIL import Image
import numpy as np

def transformation_array(image_array):
    result_array = []
    for i in range(len(image_array)):
        for j in range(len(image_array[0])):
            if image_array[i][j] != 255:
                result_array.append("1")
            else:
                result_array.append("0")
    return result_array

def process_image(img , i, path, number):
    img_gray = img.convert('L')

    result = []
    result.append(transformation_array(np.array(img_gray)))
    counter = 1
    for degree in range(-20, 21, 5):
        img_rotated = img_gray.rotate(degree, expand=False, fillcolor='white')
        img_rotated.save(path + f'{number} Class/{i}.{counter}.bmp')
        result.append(transformation_array(np.array(img_rotated)))
        counter += 1

    return result

def mixed(input_file, output_file):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    # Перемешиваем строки
    first = rows[0]
    rows = rows[1:]
    random.shuffle(rows)

    # Записываем обратно в новый CSV файл
    with open(output_file, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(first)
        writer.writerows(rows)


def shuffle_and_split_csv(input_file, test_file, training_file, test_ratio=0.8):
    with open(input_file, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    first = rows[0]
    data_rows = rows[1:]
    random.shuffle(data_rows)

    # Вычисляем количество строк для каждого файла
    split_index = int(len(data_rows) * test_ratio)
    test_rows = data_rows[:split_index]
    training_rows = data_rows[split_index:]

    # Записываем 80% строк
    with open(training_file, 'w', newline='', encoding='utf-8') as training_csv:
        writer = csv.writer(training_csv)
        writer.writerow(first)
        writer.writerows(test_rows)

    # Записываем 20% строк
    with open(test_file, 'w', newline='', encoding='utf-8') as test_csv:
        writer = csv.writer(test_csv)
        writer.writerow(first)
        writer.writerows(training_rows)


def classified(path):
    for number in range(1, 11):
        counter = 0
        for i in range(0, 10):
            image_path = path + f'{number} Class/{i}.0.bmp'
            img = Image.open(image_path)
            image_array = process_image(img, i, path, number)
            with open(
                    path + 'alldata.csv',
                    'a', newline='') as file:
                writer = csv.writer(file, delimiter=';')
                for row in image_array:
                    row.append(f"{number}")
                    row.append(f"{counter}")
                    counter += 1
                    writer.writerow(row)

if __name__ == '__main__':
    input_path = 'C:/Users/vlad/Desktop/kms_and other_trash/sucs dicks/7Semestr/ОМО/Korean/alldata.csv'
    path = 'C:/Users/vlad/Desktop/kms_and other_trash/sucs dicks/7Semestr/ОМО/Korean/'
    # with open(
    #         path + 'alldata.csv',
    #         'a', newline='') as file:
    #     writer = csv.writer(file, delimiter=',')
    #     row = []
    #     for i in range(1, 4096):
    #         row.append(f"pi{i}")
    #     writer.writerow(row)
    classified(path)
    mixed(input_path, input_path)
    shuffle_and_split_csv(input_path, '../Files/test.csv', '../Files/training.csv')

