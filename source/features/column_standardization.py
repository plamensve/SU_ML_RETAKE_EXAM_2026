import pandas as pd

def column_standardization(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset = dataset.copy()

    dataset.columns = (
        dataset.columns
        .str.replace(' ', '_')
        .str.lower()
    )

    return dataset