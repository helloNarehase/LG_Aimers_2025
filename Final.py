import numpy as np
import pandas as pd
import torch

from tabtransformer import train_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    
    X = train_df.drop('임신 성공 여부', axis=1)
    y = train_df['임신 성공 여부']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 학습 데이터에 대해 언더샘플링 적용
    rus = RandomUnderSampler(random_state=42)
    X_train, y_train = rus.fit_resample(X_train, y_train)

    # Feature Scaling 적용
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    
    return X_train, X_val, y_train, y_val, test_df

def main():
    train_file = "data_preprocess/train_e.csv"
    test_file = "data_preprocess/test_e.csv"
    model_output = "best_model.pth"
    epochs = 500
    batch_size = 64
    learning_rate = 1e-5
    
    print("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, test_df = preprocess_data(train_file, test_file)
    
    print("Starting training...")
    model = train_model(X_train, X_val, y_train, y_val, 
                        epochs=epochs, batch_size=batch_size, 
                        learning_rate=learning_rate)

    print("\nTraining complete!")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_output, map_location=device))
    model.eval()

    # test submit
    pred_proba = model.predict_proba(test_df)[:, 1]
    sample_submission = pd.read_csv('./Data/sample_submission.csv')
    sample_submission['probability'] = pred_proba
    sample_submission.to_csv('./submit.csv', index=False)
    print("Submission file saved as submit.csv")

if __name__ == "__main__":
    main()