import numpy as np
import pandas as pd
import torch

from tabtransformer import train_model
#from base_utils import numeric_columns, categoric_columns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def preprocess_data(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)
    
    X = train_df.drop('임신 성공 여부', axis=1)
    y = train_df['임신 성공 여부']

    # Encoding categorical features
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_encoded = X.copy()
    X_encoded[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])
    test_encoded = test_df.copy()
    test_encoded[categorical_cols] = ordinal_encoder.transform(test_df[categorical_cols])

    # Normalizing numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    X_encoded[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    test_encoded[numerical_cols] = scaler.transform(test_df[numerical_cols])

    # # RF feature select
    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train_encoded, y)
    # feature_importances = pd.Series(rf.feature_importances_, index=X_train_encoded.columns)
    # important_features = feature_importances[feature_importances > 0.01].index
    # X_train_filtered = X_train_encoded[important_features]
    # X_test_filtered = X_test_encoded[important_features]
    
    X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, test_encoded

def main():
    train_file = "join_train.csv"
    test_file = "join_test.csv"
    model_output = "ttf_model.pth"
    epochs = 500
    batch_size = 128
    learning_rate = 1e-6
    
    print("Loading and preprocessing data...")
    X_train, X_val, y_train, y_val, test_encoded = preprocess_data(train_file, test_file)
    
    print("Starting training...")
    model = train_model(X_train, X_val, y_train, y_val, 
                        epochs=epochs, batch_size=batch_size, 
                        learning_rate=learning_rate)

    print("\nTraining complete!")
    
    # Save model
    torch.save(model.state_dict(), model_output)
    print(f"Model saved as {model_output}")

    # test submit
    pred_proba = model.predict_proba(test_encoded)[:, 1]
    sample_submission = pd.read_csv('./Data/sample_submission.csv')
    sample_submission['probability'] = pred_proba
    sample_submission.to_csv('./submit.csv', index=False)
    print("Submission file saved as submit.csv")

if __name__ == "__main__":
    main()