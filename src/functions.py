def inspect_nulls(df):
    nulls = df.isna().sum()
    return nulls[nulls > 0]
