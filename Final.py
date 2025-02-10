import numpy as np
import pandas as pd
import torch

from tabtransformer import train_model
from sklearn.model_selection import train_test_split


def preprocess_data(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    
    X = train_df.drop('임신 성공 여부', axis=1)
    y = train_df['임신 성공 여부']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_val, y_train, y_val, test_df

def main():
    train_file = "data_preprocess/train_e.csv"
    test_file = "data_preprocess/test_e.csv"
    model_output = "ttf_model.pth"
    epochs = 500
    batch_size = 128
    learning_rate = 1e-6
    
    print("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, test_df = preprocess_data(train_file, test_file)
    
    print("Starting training...")
    model = train_model(X_train, X_val, y_train, y_val, 
                        epochs=epochs, batch_size=batch_size, 
                        learning_rate=learning_rate)

    print("\nTraining complete!")
    
    # Save model
    torch.save(model.state_dict(), model_output)
    print(f"Model saved as {model_output}")

    # test submit
    pred_proba = model.predict_proba(test_df)[:, 1]
    sample_submission = pd.read_csv('./Data/sample_submission.csv')
    sample_submission['probability'] = pred_proba
    sample_submission.to_csv('./submit.csv', index=False)
    print("Submission file saved as submit.csv")

if __name__ == "__main__":
    main()