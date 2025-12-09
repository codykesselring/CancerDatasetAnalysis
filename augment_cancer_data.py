"""
Data augmentation script for cancer.csv dataset.
Creates synthetic instances using SMOTE to reach at least 1000 total instances.
Sourced from https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ 
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load the cancer dataset."""
    df = pd.read_csv(filepath)
    print(f"Original dataset shape: {df.shape}")

    # Drop columns that are completely empty
    df = df.dropna(axis=1, how='all')

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        # Fill numeric columns with mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df


def smote_augmentation(df, target_count):
    """
    Use SMOTE to generate synthetic samples.

    Args:
        df: DataFrame with original data
        target_count: Target total number of samples
    """

    # Separate features and target
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']

    # Convert diagnosis to numeric (M=1, B=0)
    y_numeric = (y == 'M').astype(int)

    # Calculate how many samples needed per class
    current_counts = y.value_counts()
    samples_needed = target_count - len(df)

    # Use SMOTE
    min_class_count = min(current_counts)
    k_neighbors = min(5, min_class_count - 1)

    # Calculate sampling strategy to balance or reach target
    sampling_strategy = {
        0: max(current_counts[0], target_count // 2) if 'B' in current_counts.index else target_count // 2,
        1: max(current_counts[1], target_count // 2) if 'M' in current_counts.index else target_count // 2
    }

    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_numeric)

    # Create new dataframe
    augmented_df = pd.DataFrame(X_resampled, columns=X.columns)
    augmented_df.insert(0, 'diagnosis', ['M' if y == 1 else 'B' for y in y_resampled])

    # Generate IDs
    ids = []
    for i in range(len(augmented_df)):
        if i < len(df):
            ids.append(df.iloc[i]['id'])
        else:
            ids.append(f'SMOTE_{i - len(df):06d}')
    augmented_df.insert(0, 'id', ids)

    return augmented_df




def save_augmented_data(df, output_path):
    """Save the augmented dataset."""
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    INPUT_FILE = "input_data/cancer.csv"
    OUTPUT_FILE = "input_data/cancer_augmented.csv"
    TARGET_COUNT = 1000

    df_original = load_data(INPUT_FILE)

    df_augmented = smote_augmentation(df_original, TARGET_COUNT)

    save_augmented_data(df_augmented, OUTPUT_FILE)
