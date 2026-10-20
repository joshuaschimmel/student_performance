import argparse
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="Path to location of training data.")
    args = parser.parse_args()
    return args

def get_data(path):
    return pd.read_csv(path)
    
    
def train_model(X, y):
    return LinearRegression().fit(X, y)
    
def eval_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_hat)
    r2 = r2_score(y_test, y_hat)
    print(f"MSE: {mse:.3f}")
    print(f"R2 {r2:.3f}")

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    # ax.scatter(y_test, y_hat, label="accuracy")
    ax.scatter(y_hat, y_test, label="accuracy")
    ax.plot([0, 100], [0, 100], '-', color="tab:orange", label="optimum")

    ax.set(xlabel="Prediction", ylabel="Target")
    ax.legend()
    fig.suptitle("LinearRegression")
    plt.close(fig)
    mlflow.log_figure(fig, "lin_reg-plot.png")
    
    return {"mse": mse, "r2": r2}
    
def main():
    
    args = parse_args()
    
    experiment_name = 'student_performance'
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        
        df = get_data(args.data)
        X = df[["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced"]]
        y = df[["Performance Index"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        lin_reg = train_model(X_train, y_train)
        scores = eval_model(lin_reg, X_test,  y_test)
        
        print("scores: ")
        print(scores)
        

if(__name__ == "__main__"):
    main()
    