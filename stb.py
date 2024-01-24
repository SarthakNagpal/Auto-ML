import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, recall_score, precision_score, f1_score
import numpy as np

# Function to perform regression with various algorithms
def perform_regression(X_train, X_test, y_train, y_test):
    algorithms = {
        "Random Forest": RandomForestRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "ElasticNet Regression": ElasticNet()
    }

    results = {}
    for algo_name, model in algorithms.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results[algo_name] = [mse, rmse, mae, r2]

    return results

# Function to perform classification with various algorithms
def perform_classification(X_train, X_test, y_train, y_test):
    algorithms = {
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(),
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

    results = {}
    for algo_name, model in algorithms.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[algo_name] = [accuracy, recall, precision, f1]

    return results

# Function to display model performance for multiple algorithms
def display_performance(results, task_type):
    st.subheader("Model Performance:")
    for algo_name, metrics in results.items():
        st.write(f"Algorithm: {algo_name}")
        if task_type == 'Regression':
            st.write("Mean Squared Error:", metrics[0])
            st.write("Root Mean Squared Error:", metrics[1])
            st.write("Mean Absolute Error:", metrics[2])
            st.write("R-squared Score:", metrics[3])
        elif task_type == 'Classification':
            st.write("Accuracy:", metrics[0])
            st.write("Recall:", metrics[1])
            st.write("Precision:", metrics[2])
            st.write("F1 Score:", metrics[3])
        st.write("-------------------")

# Main Streamlit app
def main():
    st.title("Machine Learning Analysis with scikit-learn")

    # Warning message
    st.warning("This app does not handle outliers, null values, and non-encoded data. Please preprocess your data using Wired Whisperer before uploading.")

    # Step 1: Upload Data
    data_set = st.file_uploader('Step 1: Upload Your Dataset')
    
    if data_set is not None:
        df = pd.read_csv(data_set)
        st.success("Data uploaded successfully!")
        st.dataframe(df.head())

        # Step 2: Choose Task Type
        task_type = st.radio('Step 2: Choose the Type of ML model', ['Regression', 'Classification'])

        # Step 3: Select Target Column
        target_column = st.selectbox("Step 3: Select the Target Column", df.columns)

        # Step 4: Perform ML Analysis
        if st.button('Step 4: Perform ML Analysis') and target_column:
            # Split data into features and target
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if task_type == 'Regression':
                st.subheader("Regression Task")
                results = perform_regression(X_train, X_test, y_train, y_test)
                display_performance(results, task_type)

            elif task_type == 'Classification':
                st.subheader("Classification Task")
                results = perform_classification(X_train, X_test, y_train, y_test)
                display_performance(results, task_type)

if __name__ == "__main__":
    main()
