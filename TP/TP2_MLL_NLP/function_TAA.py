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
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, StratifiedGroupKFold, KFold, TimeSeriesSplit, GroupKFold, GroupShuffleSplit, train_test_split, LeaveOneOut, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, accuracy_score, roc_curve, precision_recall_curve, auc, f1_score, recall_score, precision_score

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek

from collections import Counter

import re

import nltk
from nltk.corpus import stopwords
# nltk.download()

def load_data(file_path, sep='\t', header=None, names=None):
    """
    Load dataset from a CSV file.
    """
    df = pd.read_csv(file_path, sep=sep, names=names)
    print(df)
    return df

    




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
    class_percentages = df[target_column].value_counts(normalize=True) * 100
    print(f"\nComptage des occurrences des classes pour '{target_column}':\n")
    print(class_counts)
    print(f"\nPourcentage des classes pour '{target_column}':\n")
    print(class_percentages)
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

    print(f"\nLes features ont été mises à l'échelle avec {scaler_type}.\n")

    return X_train_scaled, X_test_scaled

#Grid Search sur le nombre d'arbres (n_estimators) et choix du seuil (contamination) 
def tune_isolation_forest(X_train, X_test, y_test, n_estimators_list, contamination_list):
    """fonction qui explore plusieurs valeurs pour n_estimators et contamination"""
    results = []
    for n in n_estimators_list:
        for c in contamination_list:
            # Modèle Isolation Forest
            model = IsolationForest(n_estimators=n, contamination=c, random_state=42)
            model.fit(X_train)

            # Prédictions et scores
            scores = model.decision_function(X_test)
            y_pred = [1 if score < 0 else 0 for score in scores]

            # Évaluation
            roc_auc = roc_auc_score(y_test, y_pred)
            precision, recall, _ = precision_recall_curve(y_test, scores)
            pr_auc = auc(recall, precision)

            results.append({'n_estimators': n, 'contamination': c, 'roc_auc': roc_auc, 'pr_auc': pr_auc})
    
    return pd.DataFrame(results)

# Grid Search sur le nombre de voisins (n_neighbors) et seuils 
def tune_lof(X_train, X_test, y_test, n_neighbors_list):
    results = []
    for n in n_neighbors_list:
        # Modèle LOF
        lof = LocalOutlierFactor(n_neighbors=n, novelty=True)
        lof.fit(X_train)

        # Prédictions et scores
        scores = lof.decision_function(X_test)
        y_pred = [1 if score < 0 else 0 for score in scores]

        # Évaluation
        roc_auc = roc_auc_score(y_test, y_pred)
        precision, recall, _ = precision_recall_curve(y_test, scores)
        pr_auc = auc(recall, precision)

        results.append({'n_neighbors': n, 'roc_auc': roc_auc, 'pr_auc': pr_auc})

    return pd.DataFrame(results)


#################### TP2 ####################


def convert_lowercase(df: pd.DataFrame):
    """
    Convert all string columns to lowercase.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    
    Returns:
    pd.DataFrame: The dataframe with all string columns converted to lowercase.
    """
    df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    return df


# Define a function to remove URLs from text
def remove_urls(df, text):

    # Define a regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+')

    def remove_url(text):
        return url_pattern.sub('', text)

    # Apply the function to the 'text' column and create a new column 'clean_text'
    text = text.apply(remove_url(df[text]))
    return text


def remove_non_word(df: pd.DataFrame):
    """
    Remove non-word characters from the dataset.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    
    Returns:
    pd.DataFrame: The dataframe with non-word characters removed.
    """
    df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)
    return df

def remove_digits(df: pd.DataFrame):
    """
    Remove digits from the dataset.
    
    Parameters:
    df (pd.DataFrame): The dataframe containing the dataset.
    
    Returns:
    pd.DataFrame: The dataframe with digits removed.
    """
    df = df.replace(to_replace=r'\d', value='', regex=True)
    return df

def remove_stopwords(text):
    """
    Remove stopwords from the text.
    
    Parameters:
    text (str): The text to process.
    
    Returns:
    str: The text with stopwords removed.
    """
    # Tokenize the text
    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]

    # Join the words back into a single string
    return ' '.join(words)


def clean_text(df: pd.DataFrame, text_column):
    """

    """
    df = convert_lowercase(df)

    #df[text_column] = remove_urls(df, df[text_column])

    df = remove_non_word(df)

    df = remove_digits(df)

    df[text_column] = df[text_column].apply(remove_stopwords)

    return df



#################### Fonctions à améliorer ####################


def choose_scaler(scaler_type='StandardScaler'):
    """
    Specified scaler.
    
    Parameters:
    scaler_type (str): The type of scaler to use ('StandardScaler', 'RobustScaler', 'MinMaxScaler').
    """
    if scaler_type == 'StandardScaler':
        return StandardScaler()
    elif scaler_type == 'RobustScaler':
        return RobustScaler()
    elif scaler_type == 'MinMaxScaler':
        return MinMaxScaler()
    else:
        raise ValueError("Invalid scaler_type. Choose 'StandardScaler', 'RobustScaler', or 'MinMaxScaler'.")

    print(f"Le type de scaler choisis est {scaler_type}.")
    print(f"Les données n'ont pas encore été scalées mais le scaler {scaler_type} est prêt à être utilisé dans la pipeline.")



def choose_model(model_type='RandomForestClassifier', random_state=42):
    """
    Choose the model to use for classification.
    
    Parameters:
    model_type (str): The type of model to use ('RandomForestClassifier', 'LogisticRegression').
    random_state (int): Controls the shuffling applied to the data before applying the split.
    """
    if model_type == 'RandomForestClassifier':
        return RandomForestClassifier(random_state=random_state)
    elif model_type == 'LogisticRegression':
        return LogisticRegression(random_state=random_state)
    else:
        raise ValueError("Invalid model_type. Choose 'RandomForestClassifier' or 'LogisticRegression'.")

    print(f"Le modèle choisi est {model_type}.")
    print(f"Les données n'ont pas encore été procéssées mais le modèle {model_type} est prêt à être utilisé dans la pipeline.")




def choose_splitter_train_test(splitter_type, X, y, n_splits=5, test_size=0.2, random_state=42):
    """
    Choose the splitter to use for splitting the dataset into training and testing sets.
    
    Parameters:
    splitter_type (str): The type of splitter to use ('train_test_split', 'stratified_shuffle_split', 'kfold', 'stratified_kfold', 'leave_one_out', 'group_kfold', 'time_series_split').
    
    Returns:
    str: The type of splitter to use.
    """
    if splitter_type == 'train_test_split':
        splitter = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Le dataset sera divisé en training et testing sets en utilisant 'train_test_split'.")
    elif splitter_type == 'stratified_shuffle_split':
        splitter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        print("Le dataset sera divisé en training et testing sets en utilisant 'stratified_shuffle_split'.")
    elif splitter_type == 'kfold':
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        print("Le dataset sera divisé en training et testing sets en utilisant 'kfold'.")
    elif splitter_type == 'stratified_kfold':
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        print("Le dataset sera divisé en training et testing sets en utilisant 'stratified_kfold'.")
    elif splitter_type == 'leave_one_out':
        splitter = LeaveOneOut()
        print("Le dataset sera divisé en training et testing sets en utilisant 'leave_one_out'.")
    elif splitter_type == 'group_kfold':
        splitter = GroupKFold(n_splits=n_splits)
        print("Le dataset sera divisé en training et testing sets en utilisant 'group_kfold'.")
    elif splitter_type == 'time_series_split':
        splitter = TimeSeriesSplit(n_splits=n_splits)
        print("Le dataset sera divisé en training et testing sets en utilisant 'time_series_split'.")
    else:
        raise ValueError("Invalid splitter_type. Choose from 'train_test_split', 'stratified_shuffle_split', 'kfold', 'stratified_kfold', 'leave_one_out', 'group_kfold', or 'time_series_split'.")

    return splitter


    

def process_and_evaluate(X, y, scaler_type='StandardScaler', model_type='RandomForestClassifier', n_splits=5):
    """
    Build a pipeline for preprocessing the data, splitting it into training and testing sets, scaling the features, and training a model.
    
    Parameters:
    X (pd.DataFrame): The dataframe containing the features.
    y (pd.Series): The series containing the target.
    scaler_type (str): The type of scaler to use ('StandardScaler', 'RobustScaler', 'MinMaxScaler').
    splitter_type (str): The type of splitter to use ('train_test_split', 'stratified_shuffle_split', 'kfold', 'stratified_kfold', 'leave_one_out', 'group_kfold', 'time_series_split').
    model (object): The model object to use for classification.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    
    Returns:
    object: The trained model.
    """

    # Choisir le scaler et le modèle
    scaler = choose_scaler(scaler_type)
    model = choose_model(model_type)
    
    # Définir le pipeline
    pipeline = Pipeline([
        ('scaling', scaler),
        ('modeling', model)
    ])
    
    # Définir un splitter pour la validation croisée
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Évaluer le pipeline avec plusieurs métriques
    scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    results = cross_validate(pipeline, X, y, cv=splitter, scoring=scoring, return_train_score=True)
    
    # Afficher les résultats
    print("\nRésultats de la validation croisée :")
    for metric in scoring:
        print(f"Score moyen {metric} :", results[f'test_{metric}'].mean())
    
    return results














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












def apply_sampling(X_train, y_train, sampling_type):
    """
    Applique un échantillonnage (under-sampling, over-sampling, hybrid-sampling) sur l'ensemble d'entraînement
    en fonction du type d'échantillonnage choisi.
    
    Parameters:
    - X_train: Les caractéristiques d'entraînement
    - y_train: Les labels d'entraînement
    - sampling_type: Le type d'échantillonnage ('under_sampling', 'over_sampling', 'hybrid_sampling')
    
    Retourne:
    - X_train_resampled: Les caractéristiques après échantillonnage
    - y_train_resampled: Les labels après échantillonnage
    """
    # Vérifier la répartition avant équilibrage
    print("Avant échantillonnage, y_train distribution :", Counter(y_train))
    
    if sampling_type == 'under_sampling':
        # Appliquer l'under-sampling
        sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    elif sampling_type == 'over_sampling':
        # Appliquer l'over-sampling (SMOTE)
        sampler = SMOTE(random_state=42)
    elif sampling_type == 'hybrid_sampling':
        # Appliquer le hybrid-sampling (SMOTE + ENN)
        sampler = SMOTEENN(random_state=42)
    else:
        raise ValueError("sampling_type doit être 'under_sampling', 'over_sampling', ou 'hybrid_sampling'.")
    
    # Appliquer l'échantillonnage
    X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
    
    # Vérifier la répartition après échantillonnage
    print(f"Après {sampling_type}, y_train distribution :", Counter(y_train_resampled), "\n")
    
    return X_train_resampled, y_train_resampled

def train_and_predict(model, X_train, y_train, X_test):
    """
    Entraîne un modèle et effectue des prédictions.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    return y_pred, y_proba

def evaluate_performance(y_test, y_pred, y_proba=None):
    """
    Évalue les performances d'un modèle.
    """
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC: {roc_auc:.4f}")
        return roc_auc
    
def plot_confusion_matrix(y_test, y_pred):
    """
    Affiche la matrice de confusion en utilisant seaborn pour l'affichage.
    """
    cm_matrix = cm(y_test, y_pred) 
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_curves(y_test, y_proba, model_name):
    """
    Affiche les courbes ROC et Precision-Recall.
    """
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    
    # Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_superposed_curves(y_test, y_probas, model_names):
    """
    Superpose les courbes ROC et Precision-Recall de plusieurs modèles sur un même graphique.
    """
    # Courbe ROC
    plt.figure(figsize=(12, 6))

    # Tracer la courbe ROC pour chaque modèle
    for model_name, y_proba in y_probas.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_roc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC-ROC = {auc_roc:.4f})")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Superposition des courbes ROC")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Courbe Precision-Recall
    plt.figure(figsize=(12, 6))
    
    # Tracer la courbe Precision-Recall pour chaque modèle
    for model_name, y_proba in y_probas.items():
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        plt.plot(recall, precision, label=f"{model_name}")
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Superposition des courbes Precision-Recall")
    plt.legend()
    plt.grid(True)
    plt.show()

def run_pipeline(sampling_type, X_train, y_train, X_test, y_test, models):
    """
    Exécute le pipeline pour plusieurs modèles, applique le resampling, et superpose les courbes.
    """
    results = {}
    y_probas = {}  # Dictionnaire pour stocker les probabilités
    
    print(f"--- {sampling_type} ---")

    # Appliquer l'échantillonnage (sous-échantillonnage, sur-échantillonnage ou échantillonnage hybride)
    X_train_resampled, y_train_resampled = apply_sampling(X_train, y_train, sampling_type)

    for model_name, model in models.items():
        print(f"--- {model_name} ---")
        
        # Entraîner et prédire
        y_pred, y_proba = train_and_predict(model, X_train_resampled, y_train_resampled, X_test)
        
        # Évaluer les performances
        plot_confusion_matrix(y_test, y_pred)
        auc_roc = evaluate_performance(y_test, y_pred, y_proba)
        
        # Afficher les courbes
        if y_proba is not None:
            plot_curves(y_test, y_proba, model_name)
            
        # Stocker les probabilités pour la courbe ROC et Precision-Recall
        y_probas[model_name] = y_proba
        
        results[model_name] = {"AUC-ROC": auc_roc}
    
    # Afficher les courbes superposées
    plot_superposed_curves(y_test, y_probas, models.keys())
    
    return results

#run_pipeline(X_train, y_train, X_test, y_test, {
#    "Logistic Regression": LogisticRegression(),
#    "Random Forest": RandomForestClassifier()
#})

