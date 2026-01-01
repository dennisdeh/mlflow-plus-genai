import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 1. Connect to MLflow running in Docker
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Titanic_Logistic_Regression")


def train_titanic():
    # 2. Load the dataset (using a public URL for simplicity)
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)

    # 3. Simple Preprocessing
    # Drop columns that aren't useful for a simple model
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    # Convert categorical variables to dummy variables
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Start MLflow Run
    with mlflow.start_run():
        params = {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # Predict and calculate metrics
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        # 5. Log parameters, metrics, and model
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Log the model
        mlflow.sklearn.log_model(model, name="logistic-regression-model")

        print(f"Model trained with accuracy: {accuracy:.4f}")
        print("Run logged to MLflow UI at http://localhost:5000")


if __name__ == "__main__":
    train_titanic()