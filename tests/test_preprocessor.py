import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import HeartDiseasePreprocessor

@pytest.fixture
def preprocessor():
    return HeartDiseasePreprocessor()

@pytest.fixture
def sample_train_data():
    return pd.DataFrame({
        'age': [30, 40, 50, 60],
        'sex': [1, 0, 1, 0],
        'cp': [1, 2, 3, 4],
        'trestbps': [110, 120, 130, 140],
        'chol': [180, 200, 220, 240],
        'fbs': [0, 1, 0, 1],
        'restecg': [0, 1, 2, 0],
        'thalach': [140, 150, 160, 170],
        'exang': [0, 1, 0, 1],
        'oldpeak': [0.0, 1.0, 2.0, 3.0],
        'slope': [1, 2, 3, 1],
        'ca': [0, 1, 2, 3],
        'thal': [3, 6, 7, 3]
    })

def test_fit_transform_determinism(preprocessor, sample_train_data):
    # Ensure that fitting and transforming the same data twice yields identical results
    preprocessor.fit(sample_train_data)
    transformed_1 = preprocessor.transform(sample_train_data)
    
    preprocessor_2 = HeartDiseasePreprocessor()
    preprocessor_2.fit(sample_train_data)
    transformed_2 = preprocessor_2.transform(sample_train_data)
    
    pd.testing.assert_frame_equal(transformed_1, transformed_2)

def test_column_ordering_robustness(preprocessor, sample_train_data):
    # Preprocessor should handle columns in any order by using self.feature_order
    preprocessor.fit(sample_train_data)
    
    shuffled_data = sample_train_data[sample_train_data.columns[::-1]]
    transformed = preprocessor.transform(shuffled_data)
    
    assert transformed.shape[0] == 4
    assert "num__age" in transformed.columns

@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_handle_unknown_categories(preprocessor, sample_train_data):
    # Preprocessor should not crash on unknown categorical labels (handle_unknown='ignore')
    preprocessor.fit(sample_train_data)
    
    unknown_data = sample_train_data.iloc[[0]].copy()
    unknown_data['cp'] = 99 # Unknown label
    
    transformed = preprocessor.transform(unknown_data)
    # The CP columns (except the ones seen in fit) should be zero
    cp_cols = [c for c in transformed.columns if "cat__cp" in c]
    assert all(transformed[cp_cols].iloc[0] == 0)

def test_scaling_consistency(preprocessor, sample_train_data):
    # Verify that num__age is correctly scaled (StandardScaler: mean=0, std=1)
    preprocessor.fit(sample_train_data)
    transformed = preprocessor.transform(sample_train_data)
    
    # Mean of 30, 40, 50, 60 is 45. Standard deviation is ~11.18
    # 30 -> (30-45)/std = -1.34
    assert pytest.approx(transformed['num__age'].mean(), abs=1e-7) == 0
    assert pytest.approx(transformed['num__age'].std(ddof=0), abs=1e-7) == 1
