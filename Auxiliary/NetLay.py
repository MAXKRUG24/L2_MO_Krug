import pandas as pd

from .Sigmoid import sigmoid


def net_lay(img_matrix: pd.DataFrame, scales_matrix: pd.DataFrame, scale_end: bool = False) -> pd.DataFrame:
    col = scales_matrix.shape[1]

    arr = []

    for j in range(col):
        var = img_matrix * scales_matrix.iloc[:, j]
        var = var.sum()
        if not scale_end:
            var = sigmoid(var)
        arr.append(round(var, 4))
    y = pd.DataFrame(arr)

    return y.T.iloc[0]
