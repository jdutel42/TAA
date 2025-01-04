import numpy as np
np.set_printoptions(threshold=10000, suppress = True)

import pandas as pd

import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPathCollection

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, roc_curve, precision_recall_curve, auc, f1_score, recall_score, precision_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from collections import Counter

def get_info(df: pd.DataFrame):
    """
    Display general information about the dataset.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    """
    print("### Infos générale ###")
    print("\n")
    print(df.info())
    print("\n")
    print("Le dataset contient {} lignes et {} colonnes.".format(df.shape[0], df.shape[1]))
    print("\n")

    print("### Description ###")
    print("\n")
    print(df.describe())
    print("\n")

    print("### Valeurs manquantes ###")
    print("\n")
    print(df.isnull().sum())
    print("\n")
    print("Le dataset contient {} valeurs manquantes.".format(df.isnull().sum().sum()))
    print("\n")

    print("### Colonnes catégorielles ###")
    print("\n")
    print("=== Nombre de colonnes catégorielles ===")
    print("\n")
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(categorical_cols)
    print("\n")
    print("Le dataset contient {} colonnes catégorielles (qu'il faudra surement réencoder numériquement).".format(len(categorical_cols)))
    print("\n")
    print("=== Nombre de modalités par colonnes catégorielles ===")
    print("\n")
    for col in categorical_cols:
        print(f"--- {col} ---")
        print(df[col].value_counts())
        print("\n")
        print(f"Nombre de modalité dans la colonne '{col}' : {len(df[col].unique())}")
        print("\n")
    print("\n")
    return None



def load_data(file_path, header=None):
    """
    Load dataset from a CSV file.
    """
    if header is None:
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(file_path, header=0)
















def preprocess_data(df, target_column):
    """
    Preprocess the data by handling missing values, encoding categorical variables, and scaling features.
    """
    # Handle missing values
    df = df.dropna()

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, label_encoders

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using various metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }