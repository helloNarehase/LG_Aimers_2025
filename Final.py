import numpy as np
import pandas as pd
import torch

from tabtransformer import train_model
from base_utils import numeric_columns, categoric_columns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path).drop(columns=['ID'])
    test_df = pd.read_csv(test_file_path).drop(columns=['ID'])
    
    X = train_df.drop('임신 성공 여부', axis=1)
    y = train_df['임신 성공 여부']

    # Encoding categorical features
    for col in categoric_columns:
        X[col] = X[col].astype(str)
        test_df[col] = test_df[col].astype(str)

    ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_encoded = X.copy()
    X_train_encoded[categoric_columns] = ordinal_encoder.fit_transform(X[categoric_columns])
    X_test_encoded = test_df.copy()
    X_test_encoded[categoric_columns] = ordinal_encoder.transform(test_df[categoric_columns])

    # Normalizing numerical features
    scaler = StandardScaler()
    X_train_encoded[numeric_columns] = scaler.fit_transform(X[numeric_columns])
    X_test_encoded[numeric_columns] = scaler.transform(test_df[numeric_columns])
    
    # RF feature select
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_encoded, y)
    feature_importances = pd.Series(rf.feature_importances_, index=X_train_encoded.columns)
    important_features = feature_importances[feature_importances > 0.01].index
    X_train_filtered = X_train_encoded[important_features]
    X_test_filtered = X_test_encoded[important_features]
    X_train, X_val, y_train, y_val = train_test_split(X_train_filtered, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, X_test_filtered

def main():
    train_file = "train_cleaned.csv"
    test_file = "test_cleaned.csv"
    model_output = "tabtransformer_model.pth"
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