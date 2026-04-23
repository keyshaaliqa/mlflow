import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
import joblib

def load_data():
    df = pd.read_csv('d:/DOCUMENTS/Exam/UTS Model Deployment/2. Scikit-Learn Pipeline/A.csv')
    target = pd.read_csv('d:/DOCUMENTS/Exam/UTS Model Deployment/2. Scikit-Learn Pipeline/A_targets.csv')
    data = df.merge(target, on='Student_ID')
    return data

def preprocessing_pipeline(data):
    numeric_features = data.select_dtypes(include=np.number).columns.tolist()
    categorical_features = data.select_dtypes(include='object').columns.tolist()

    # remove target
    categorical_features = [col for col in categorical_features if col != 'placement_status']
    numeric_features = [col for col in numeric_features if col not in ['placement_status', 'salary_lpa', 'Student_ID']]

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor, numeric_features + categorical_features

def split_data(data):
    X = data.drop(['Student_ID','placement_status','salary_lpa'], axis=1)
    y_class = data['placement_status']
    y_reg = data['salary_lpa']

    X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)
    _, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    return X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg

def train_classification(preprocessor, X_train, X_test, y_train, y_test):

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC()
    }

    best_model = None
    best_acc = 0

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            acc = accuracy_score(y_test, preds)

            # Logging MLflow
            mlflow.log_param("model", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(pipeline, "model")

            print(f"{name} Accuracy: {acc}")

            if acc > best_acc:
                best_acc = acc
                best_model = pipeline

    return best_model

def train_regression(preprocessor, X_train, X_test, y_train, y_test):

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(),
        "SVR": SVR()
    }

    best_model = None
    best_r2 = -np.inf

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            r2 = r2_score(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))

            # Logging
            mlflow.log_param("model", name)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("rmse", rmse)

            mlflow.sklearn.log_model(pipeline, "model")

            print(f"{name} R2: {r2} | RMSE: {rmse}")

            if r2 > best_r2:
                best_r2 = r2
                best_model = pipeline

    return best_model

def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":

    data = load_data()

    preprocessor, _ = preprocessing_pipeline(data)

    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = split_data(data)

    print("\n=== TRAINING CLASSIFICATION ===")
    best_cls_model = train_classification(preprocessor, X_train, X_test, y_train_cls, y_test_cls)

    print("\n=== TRAINING REGRESSION ===")
    best_reg_model = train_regression(preprocessor, X_train, X_test, y_train_reg, y_test_reg)

    # Save best models
    save_model(best_cls_model, "best_classification.pkl")
    save_model(best_reg_model, "best_regression.pkl") 