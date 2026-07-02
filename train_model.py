import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


DATA_PATH = "vgsales.csv"


def load_data(path):
    df = pd.read_csv(path)
    return df


def prepare_data(df):
    df = df.copy()

    # Keep required columns only
    columns = [
        "Name",
        "Platform",
        "Year",
        "Genre",
        "Publisher",
        "NA_Sales",
        "EU_Sales",
        "JP_Sales",
        "Other_Sales",
        "Global_Sales",
    ]

    df = df[columns]

    # Handle missing values
    numeric_cols = ["Year", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    df["Publisher"] = df["Publisher"].fillna("Unknown")
    df["Genre"] = df["Genre"].fillna("Unknown")
    df["Platform"] = df["Platform"].fillna("Unknown")
    df["Name"] = df["Name"].fillna("Unknown")

    # Similar to the report: focus on Simulation games after 2010 when available
    filtered_df = df[(df["Genre"] == "Simulation") & (df["Year"] > 2010)]

    if len(filtered_df) < 20:
        filtered_df = df

    features = ["Year", "NA_Sales", "EU_Sales", "JP_Sales"]
    target = "Global_Sales"

    X = filtered_df[features]
    y = filtered_df[target]

    return filtered_df, X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, X_test, y_test, predictions, mse, r2


def save_plot(y_test, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.7)
    plt.xlabel("Actual Global Sales")
    plt.ylabel("Predicted Global Sales")
    plt.title("Actual vs Predicted Global Sales")
    plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png")
    plt.close()


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Preparing data...")
    filtered_df, X, y = prepare_data(df)

    print("Training Linear Regression model...")
    model, X_test, y_test, predictions, mse, r2 = train_model(X, y)

    print("Saving model...")
    joblib.dump(model, "models/linear_regression_model.pkl")

    print("Saving plot...")
    save_plot(y_test, predictions)

    results = pd.DataFrame({
        "Actual_Global_Sales": y_test.values,
        "Predicted_Global_Sales": predictions
    })

    results.to_csv("outputs/predictions.csv", index=False)

    print("\nModel Evaluation")
    print("----------------")
    print(f"MSE: {mse}")
    print(f"R2 Score: {r2}")

    print("\nProject completed successfully!")


if __name__ == "__main__":
    main()
