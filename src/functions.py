import pandas as pd
import numpy as np


def inspect_nulls(df):
    nulls = df.isna().sum()
    return nulls[nulls > 0]


def to_numeric(df, columns):
    df[columns] = df[columns].apply(pd.to_numeric, errors='coerce')
    return df


def to_binary_from_nan(data, column):
    return np.where(data[column].isna(), 0, 1)
