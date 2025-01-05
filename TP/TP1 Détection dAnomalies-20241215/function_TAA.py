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
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, StratifiedGroupKFold, KFold, TimeSeriesSplit, GroupKFold, GroupShuffleSplit, train_test_split, LeaveOneOut, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, roc_curve, precision_recall_curve, auc, f1_score, recall_score, precision_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from collections import Counter


def load_data(file_path, header=None):
    """
    Load dataset from a CSV file.
    """
    if header is None:
        return pd.read_csv(file_path)
    else:
        return pd.read_csv(file_path, header=0)
    




def get_info(df: pd.DataFrame):
    """
    Display general information about the dataset.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    """
    print("### Infos générale ###\n")
    print(df.info())
    print(f"\nLe dataset contient {df.shape[0]} lignes et {df.shape[1]} colonnes.\n")

    print("### Description ###\n")
    print(df.describe().to_string())
    print("\n")

    print("### Valeurs manquantes ###\n")
    missing_values = df.isnull().sum()
    print(missing_values)
    print(f"\nLe dataset contient {missing_values.sum()} valeurs manquantes.\n")

    print("### Colonnes catégorielles ###\n")
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"=== Nombre de colonnes catégorielles: {len(categorical_cols)} ===\n")
    print(categorical_cols)
    print("\n=== Nombre de modalités par colonnes catégorielles ===\n")
    for col in categorical_cols:
        print(f"--- {col} ---")
        print(df[col].value_counts().to_string())
        print(f"\nNombre de modalité dans la colonne '{col}' : {len(df[col].unique())}\n")
    print("\n")
    return None



def plot_class_distribution(df: pd.DataFrame, target_column):
    """
    Plot the distribution of the target variable and display the count of occurrences for each class.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    target_column (str): The name of the target column.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=target_column, data=df, palette='viridis')
    plt.title(f"Distribution de la variable '{target_column}'")
    plt.xticks(rotation=90)  # Rotation des étiquettes des labels
    plt.show()
    
    class_counts = df[target_column].value_counts()
    print(f"\nComptage des occurrences des classes pour '{target_column}':\n")
    print(class_counts)
    return None



def dropna(df: pd.DataFrame):
    """
    Drop rows with missing values.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    
    Returns:
    pd.DataFrame: The dataframe with missing values dropped.
    """
    missing_values_before = df.isnull().sum()
    print(f"\nLe dataset contenanait {missing_values_before.sum()} valeurs manquantes avant l'opération.\n")

    df = df.dropna()

    missing_values_after = df.isnull().sum()
    print(f"\nLe dataset contient maintenant {missing_values_after.sum()} valeurs manquantes.\n")
    print("Les valeurs manquantes ont été supprimées.")

    return df




def encode_categorical(df: pd.DataFrame, encoding_type='LabelEncoder'):
    """
    Encode categorical variables in the dataset using the specified encoding type.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    encoding_type (str): The type of encoding to use ('LabelEncoder' or 'OneHotEncoder').
    
    Returns:
    pd.DataFrame: The dataframe with categorical variables encoded.
    dict: A dictionary containing the encoders for each column.
    """
    encoders = {}
    if encoding_type == 'LabelEncoder':
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            encoders[column] = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"Label encoding for column '{column}':")
            for label, encoded in encoders[column].items():
                print(f"  {label}: {encoded}")
            print("\n")
    elif encoding_type == 'OneHotEncoder':
        ohe = OneHotEncoder(sparse=False, drop='first')
        encoded_cols = ohe.fit_transform(df.select_dtypes(include=['object']))
        encoded_df = pd.DataFrame(encoded_cols, columns=ohe.get_feature_names_out(df.select_dtypes(include=['object']).columns))
        df = df.drop(df.select_dtypes(include=['object']).columns, axis=1)
        df = pd.concat([df, encoded_df], axis=1)
        encoders = ohe
        print("One-hot encoding applied to categorical variables.")
        for label, encoded in zip(df.select_dtypes(include=['object']).columns, ohe.get_feature_names_out(df.select_dtypes(include=['object']).columns)):
            print(f"  {label}: {encoded}")
    else:
        raise ValueError("Invalid encoding_type. Choose 'LabelEncoder' or 'OneHotEncoder'.")

    print("Les variables catégorielles ont été encodées.")

    return df, encoders





def split_feature_label(df: pd.DataFrame, target_column):
    """
    Split the dataset into features and target.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    target_column (str): The name of the target column.
    
    Returns:
    pd.DataFrame: The dataframe containing the features.
    pd.Series: The series containing the target.
    """
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    print(f"Les features sont dans le dataframe X ({X.shape[0]} lignes, {X.shape[1]} colonnes).")
    print(f"La target est dans la série y ({y.shape[0]} éléments).")

    return X, y



def split_train_test(X: pd.DataFrame, y: pd.Series, method='train_test_split', test_size=0.2, random_state=42, n_splits=5, groups=None):
    """
    Split the dataset into training and testing sets using various methods.
    
    Parameters:
    X (pd.DataFrame): The dataframe containing the features.
    y (pd.Series): The series containing the target.
    method (str): The method to use for splitting ('train_test_split', 'stratified_shuffle_split', 'kfold', 'stratified_kfold', 'leave_one_out', 'group_kfold', 'time_series_split').
    test_size (float): The proportion of the dataset to include in the test split (only for 'train_test_split' and 'stratified_shuffle_split').
    random_state (int): Controls the shuffling applied to the data before applying the split.
    n_splits (int): Number of splits (only for 'kfold', 'stratified_kfold', 'group_kfold', 'time_series_split').
    groups (array-like): Group labels for the samples used while splitting the dataset into train/test set (only for 'group_kfold').
    
    Returns:
    list: A list containing the training and testing features and target.
    """
    if method == 'train_test_split':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return [(X_train, X_test, y_train, y_test)]
    
    elif method == 'stratified_shuffle_split':
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in sss.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        return [(X_train, X_test, y_train, y_test)]
    
    elif method == 'kfold':
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))
        return splits
    
    elif method == 'stratified_kfold':
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))

        print(f"Le dataset a bien été divisé en {n_splits} folds, avec pour chaque fold, X_train, X_test, y_train, y_test.")

        print("split[0], split[1], split[2], split[3] are folds")
        print("split[0][0] is the X_train set of the first fold,")
        print("split[0][1] is the X_test set of the first fold,")
        print("split[0][2] is the y_train set of the first fold,")
        print("split[0][3] is the y_test set of the first fold")
        print("etc...")

        return splits
    
    elif method == 'leave_one_out':
        loo = LeaveOneOut()
        splits = []
        for train_index, test_index in loo.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))
        return splits
    
    elif method == 'group_kfold':
        gkf = GroupKFold(n_splits=n_splits)
        splits = []
        for train_index, test_index in gkf.split(X, y, groups=groups):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))
        return splits
    
    elif method == 'time_series_split':
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            splits.append((X_train, X_test, y_train, y_test))
        return splits
    
    else:
        raise ValueError("Invalid method. Choose from 'train_test_split', 'stratified_shuffle_split', 'kfold', 'stratified_kfold', 'leave_one_out', 'group_kfold', or 'time_series_split'.")



def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame, scaler_type='StandardScaler'):
    """
    Scale the features in the dataset using the specified scaler.
    
    Parameters:
    X_train (pd.DataFrame): The dataframe containing the features for train data.
    X_test (pd.DataFrame): The dataframe containing the features for test data.
    scaler_type (str): The type of scaler to use ('StandardScaler', 'RobustScaler', 'MinMaxScaler').
    
    Returns:
    pd.DataFrame: The scaled training features.
    pd.DataFrame: The scaled testing features.
    """
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Invalid scaler_type. Choose 'StandardScaler', 'RobustScaler', or 'MinMaxScaler'.")

    # Applique le scaling sur X_train, et utilise le même scaler pour transformer X_test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Les features ont été mises à l'échelle avec {scaler_type}.")

    return X_train_scaled, X_test_scaled













def preprocess_data(df, target_column):
    """
    Preprocess the data by encoding categorical variables, and scaling features.
    """
    # Display general information about the dataset
    get_info(df)

    # Plot the distribution of the target variable
    plot_class_distribution(df, target_column)

    # Drop rows with missing values
    df = dropna(df)

    # Encode categorical variables
    df, encoders = encode_categorical(df, encoding_type='OneHotEncoder')

    # Split the dataset into features and target
    X, y = split_feature_label(df, target_column)

    # Split the dataset into training and testing sets
    splits = split_train_test(X, y, method='train_test_split', test_size=0.2, random_state=42)

    # Scale the features
    X_train_scaled, X_test_scaled = scale_features(splits[0][0], splits[0][1], scaler_type='StandardScaler')

    return X_train_scaled, X_test_scaled, splits[0][2], splits[0][3], encoders






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