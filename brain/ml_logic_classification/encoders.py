import pandas as pd


def tumor_encoded(df) :
    mapping = {
        'notumor': 0,
        'meningioma': 1,
        'glioma': 2,
        'pituitary': 3}

    df["tumor_type_encoded"] = df["tumor_type"].map(mapping)
    return df
