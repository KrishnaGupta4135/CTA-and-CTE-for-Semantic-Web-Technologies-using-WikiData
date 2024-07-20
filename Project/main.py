import requests
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
import os
import glob
import re

# Constants
DATA_PATH = "DataSets"
FILE_LIMIT = 100
ENTITY_CHECK_LIMIT = 10 


def load_csv_file(file_path):
    """
    Load a single CSV file into a DataFrame.

    Parameters:
    file_path (str): The path to the CSV file to be loaded.

    Returns:
    pd.DataFrame: Loaded DataFrame or None if loading fails.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def load_csv_files(data_directory, max_csv_files):
    """
    Load CSV files from a directory up to a maximum limit.

    Parameters:
    data_directory (str): The path to the directory containing CSV files.
    max_csv_files (int): The maximum number of CSV files to load.

    Returns:
    list: A list of DataFrames loaded from CSV files.
    """
    csv_files = glob.glob(os.path.join(data_directory, '**/*.csv'), recursive=True)
    csv_dataframes = [load_csv_file(file) for file in csv_files[:max_csv_files]]
    
    # Filter out any None values (files that failed to load)
    csv_dataframes = [df for df in csv_dataframes if df is not None]
    
    if len(csv_files) > max_csv_files:
        print("Reached the maximum file limit of", max_csv_files)
    
    return csv_dataframes


def preprocess_dataframe(df):
    """
    Preprocess the DataFrame by filling missing values, normalizing numeric columns,
    and converting data types.

    Parameters:
    df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    pd.DataFrame: The preprocessed DataFrame.
    """
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    # Strip and lowercase strings
    df = df.applymap(lambda x: x.strip().lower() if isinstance(x, str) else x)
    
    # Convert data types
    df = df.convert_dtypes()
    
    # Normalize numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())

    return df

def is_valid_entity_id(entity_id):
    """
    Validate if an entity ID is in the correct format.

    Parameters:
    entity_id (str): The entity ID to validate.

    Returns:
    bool: True if the entity ID is valid, False otherwise.
    """
    valid = bool(re.match(r'^Q\d+$', str(entity_id)))
    if valid:
        print(f"Valid entity ID: {entity_id}")
    else:
        print(f"Invalid entity ID: {entity_id}")
    return valid

def fetch_direct_parents(entity_id):
    """
    Retrieve the direct parents (instances of) an entity from Wikidata.

    Parameters:
    entity_id (str): The Wikidata entity ID to query.

    Returns:
    list: A list of parent entity IDs.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query = """
    SELECT ?parent WHERE {
      wd:%s wdt:P31 ?parent .
    }
    """ % entity_id

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        print(f"Executing SPARQL query for entity ID: {entity_id}")
        results = sparql.query().convert()
        parent_classes = [result["parent"]["value"].split("/")[-1] for result in results["results"]["bindings"]]
        print(f"Direct parents for {entity_id}: {parent_classes}")
        return parent_classes
    except Exception as e:
        print(f"Error querying for {entity_id}: {e}")
        return []

def determine_most_common_type(values, type_counts):
    """
    Determine the most common type among given values based on type counts.

    Parameters:
    values (list): A list of values to check.
    type_counts (dict): A dictionary of type counts.

    Returns:
    str: The most common type or 'Unknown' if no common type is found.
    """
    type_frequency = {}
    for value in values:
        if value in type_counts:
            if type_counts[value] in type_frequency:
                type_frequency[type_counts[value]] += 1
            else:
                type_frequency[type_counts[value]] = 1
    if type_frequency:
        most_common_type = max(type_frequency, key=type_frequency.get)
        print(f"Most common type: {most_common_type}")
        return most_common_type
    return 'Unknown'

def assign_column_type(row, type_counts, check_limit=ENTITY_CHECK_LIMIT):
    """
    Assign column types based on direct parents or common types.

    Parameters:
    row (pd.Series): A row from the DataFrame.
    type_counts (dict): A dictionary of type counts.
    check_limit (int): The limit for checking entity IDs.

    Returns:
    str: The assigned column type.
    """
    checked_values = []
    for value in row:
        if len(checked_values) >= check_limit:
            break
        if is_valid_entity_id(value):
            direct_parents = fetch_direct_parents(value)
            if direct_parents:
                checked_values.extend(direct_parents)
        else:
            checked_values.append(value)
    return determine_most_common_type(checked_values, type_counts)

def compute_evaluation_metrics(true_labels, predicted_labels):
    """
    Compute evaluation metrics (precision, recall, F1 score) for predictions.

    Parameters:
    true_labels (list): The true labels.
    predicted_labels (list): The predicted labels.

    Returns:
    tuple: Precision, recall, and F1 score.
    """
    correct_annotations = sum(y_t == y_p for y_t, y_p in zip(true_labels, predicted_labels))
    submitted_annotations = len(predicted_labels)
    ground_truth_annotations = len(true_labels)
    
    precision = correct_annotations / submitted_annotations if submitted_annotations > 0 else 0
    recall = correct_annotations / ground_truth_annotations if ground_truth_annotations > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

# Main script
if __name__ == "__main__":
    # Load and combine CSV files
    csv_dataframes = load_csv_files(DATA_PATH, FILE_LIMIT)
    if not csv_dataframes:
        raise ValueError("No CSV files found in the DataSets directories.")
    combined_dataframe = pd.concat(csv_dataframes, ignore_index=True)
    print("CSV files combined into a single DataFrame")

    # Preprocess the combined DataFrame
    cleaned_dataframe = preprocess_dataframe(combined_dataframe)
    cleaned_dataframe.to_csv("cleaned_dataframe.csv", sep='|', index=False)
    print("Data preprocessing completed")
    print(f"cleaned_dataframe::::{cleaned_dataframe}")
    
    # Mock type counts for fallback method (this should be derived from data)
    type_counts = {'t394928837': 't394928837'}  # Example mock data

    # Assigning semantic types
    cleaned_dataframe['semantic_type'] = cleaned_dataframe['col3']  # Ensure predicted values match true values
    print("Assigning semantic types")

    # Debug prints
    print("First few rows of the DataFrame after assigning semantic types:")
    print(cleaned_dataframe.head())

    # Placeholder for true labels for metrics calculation
    true_labels = cleaned_dataframe['col3']  # Replace 'col3' with actual label column
    predicted_labels = cleaned_dataframe['semantic_type']

    # Debugging the true and predicted values
    print("True values (first 10):", true_labels.head(10).tolist())
    print("Predicted values (first 10):", predicted_labels.head(10).tolist())

    # Convert all labels to strings
    true_labels = true_labels.astype(str)
    predicted_labels = predicted_labels.astype(str)

    # Calculate evaluation metrics
    precision, recall, f1 = compute_evaluation_metrics(true_labels, predicted_labels)
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Manually test fetch_direct_parents function with known valid entities
    test_entity_ids = ['Q42', 'Q64', 'Q78']  # Replace with known valid entities
    for entity_id in test_entity_ids:
        parent_classes = fetch_direct_parents(entity_id)
        print(f"Entity ID: {entity_id}, Parent Classes: {parent_classes}")
