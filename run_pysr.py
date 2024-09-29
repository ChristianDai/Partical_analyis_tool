import argparse
import pandas as pd
import pickle
from pysr import PySRRegressor

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X, y

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(model, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def main():
    parser = argparse.ArgumentParser(description='Train a PySR model on given data.')
    parser.add_argument('data_path', type=str, help='Path to the training data CSV file')
    parser.add_argument('model_path', type=str, help='Path to save/load the pickled PySR model')
    args = parser.parse_args()

    # Load data
    X, y = load_data(args.data_path)
    # Load or initialize model
    model = load_model(args.model_path)

    # Fit model
    model.fit(X, y)

    # Save model
    save_model(model, args.model_path)

    print(f"Model trained and saved to {args.model_path}")

if __name__ == "__main__":
    main()