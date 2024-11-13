import pandas as pd
import numpy as np


def softmax(series):
    exp_values = np.exp(series - np.max(series))  # для предотвращения переполнения
    return exp_values / exp_values.sum()


if __name__ == '__main__':
    # Пример использования:
    data = [1, 11, 5]
    df = pd.DataFrame(data)

    df = softmax(df)

    print(df)
