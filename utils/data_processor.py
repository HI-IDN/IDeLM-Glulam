# utils/data_processor.py
import pandas as pd


def filter_glulam_data(file_path, depth):
    df = pd.read_csv(file_path)
    available_depths = df['depth'].unique()
    assert depth in available_depths, f"Depth {depth} mm not found. Available depths are: {available_depths}"
    return df[df['depth'] == depth]
