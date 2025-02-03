import argparse
import pandas as pd
import torch
from tabtransformer import train_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def preprocess_data(train_file_path, test_file_path):
    train_df = pd.read_csv(train_file_path, encoding='utf-8-sig')
    test_df = pd.read_csv(test_file_path, encoding='utf-8-sig')
    
    # Handling categorical features
    categorical_cols = train_df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.fit_transform(test_df[col])
        label_encoders[col] = le
    
    # Normalizing numerical features
    # numerical_columns = train_df.select_dtypes(include=['int64', 'float64']).columns
    # scaler = StandardScaler()
    # train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])
    # test_df[numerical_columns] = scaler.fit_transform(test_df[numerical_columns])
    
    X = train_df.drop(["임신 성공 여부"], axis=1)
    y = train_df["임신 성공 여부"]

    # PCA
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X)
    test_scaled = scaler.transform(test_df)

    pca = PCA(n_components=0.95)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    X_train, X_val, y_train, y_val = train_test_split(train_pca, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val, test_pca

def main():
    train_file = "train_cleaned.csv"
    test_file = "test_cleaned.csv"
    model_output = "tabtransformer_model.pth"
    epochs = 200
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

if __name__ == "__main__":
    main()