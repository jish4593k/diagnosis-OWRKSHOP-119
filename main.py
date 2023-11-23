import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import spacy
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox

# Load Data
def load_data():
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        return df
    return None

# Preprocessing
def preprocess_data(df):
    df = df.drop(['Season'], axis=1)
    group_bool = ['Childish diseases', 'Accident or serious trauma', 'Surgical intervention']
    df[group_bool] = df[group_bool].apply(lambda x: pd.Series(x).map({'yes': 1, 'no': 0}))
    df = pd.get_dummies(df.drop(['Diagnosis'], axis=1))
    return df

# Multiple Regression
def perform_multiple_regression(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model

# Data Mining (KMeans Clustering)
def perform_kmeans_clustering(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_scaled)
    return kmeans.labels_

# Model Testing
def test_model(model, X_test, Y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(Y_test, predictions)
    return mse

# GUI
def show_gui():
    root = tk.Tk()
    root.title("Data Mining and Analysis Tool")

    # Load Data Button
    load_data_button = tk.Button(root, text="Load Data", command=load_and_preprocess_data)
    load_data_button.pack(pady=10)

    # Multiple Regression Button
    regression_button = tk.Button(root, text="Perform Multiple Regression", command=perform_and_display_regression)
    regression_button.pack(pady=10)

    # KMeans Clustering Button
    clustering_button = tk.Button(root, text="Perform KMeans Clustering", command=perform_and_display_clustering)
    clustering_button.pack(pady=10)

    root.mainloop()

# Load and Preprocess Data
def load_and_preprocess_data():
    global df
    df = load_data()
    if df is not None:
        df = preprocess_data(df)
        messagebox.showinfo("Success", "Data loaded and preprocessed successfully!")

# Perform and Display Multiple Regression
def perform_and_display_regression():
    if 'Diagnosis' not in df.columns:
        messagebox.showerror("Error", "Diagnosis column not found. Please load and preprocess the data first.")
        return

    X = df.drop(['Diagnosis'], axis=1)
    Y = df['Diagnosis']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    model = perform_multiple_regression(X_train, Y_train)
    mse = test_model(model, X_test, Y_test)

    messagebox.showinfo("Multiple Regression Results", f"Multiple Regression Mean Squared Error: {mse:.4f}")

def perform_and_display_clustering():
    if 'Diagnosis' not in df.columns:
        messagebox.showerror("Error", "Diagnosis column not found. Please load and preprocess the data first.")
        return

    X = df.drop(['Diagnosis'], axis=1)
    labels = perform_kmeans_clustering(X)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Age', y='Number of hours spent sitting per day', hue=labels, data=df, palette='viridis')
    plt.title('KMeans Clustering Results')
    plt.show()

if __name__ == "__main__":
    df = pd.DataFrame()  # Global variable to store the loaded data

    spacy_model = spacy.load("en_core_web_sm")

    # GUI
    show_gui()



