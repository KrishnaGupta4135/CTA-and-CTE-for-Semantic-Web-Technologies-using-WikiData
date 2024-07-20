# Advanced Column-Type Annotation (CTA) and Column-Type Entity for Semantic Web Technologies using WikiData

## Introduction

Knowledge Graph (KG) is a robust framework for representing knowledge that arranges data in an orderly fashion to promote knowledge discovery and semantic comprehension. A KG represents data as entities (nodes) and relationships (edges) between them, creating a structure like to a graph, in contrast to conventional databases or flat data structures. KGs record intricate links between elements, allowing rich contextual information and semantic relationships to be represented.

In this project, we look into cell-entity annotation (CEA) and column-type annotation (CTA), two important approaches for improving semantic table interpretation by mapping data cells and columns to pertinent entities and kinds in a Knowledge Graph. These techniques help with data integration, entity resolution, and semantic search, which improves the usefulness and comprehension of tabular data in a variety of applications.

## Approach and Methodology

This project aims to map tabular data cells and columns to relevant entities and types in a KG, thereby enabling semantic table interpretation. The approach includes the following steps:

1. **Preprocessing**: The data preprocessing step includes handling missing values, normalizing numeric columns, converting data types, and string manipulation (stripping and lowercasing).
2. **Entity Validation**: Ensuring that entity IDs are in the correct format and validating them using a predefined pattern.
3. **SPARQL Querying**: Using the SPARQL endpoint to query WikiData for direct parent entities.
4. **Type Determination**: Determining the most common type among given values based on type counts.
5. **Evaluation**: Computing evaluation metrics (precision, recall, F1 score) for predictions.

## Script Overview

### Functions

- **load_csv_file**: Load a single CSV file into a DataFrame.
- **load_csv_files**: Load multiple CSV files from a directory up to a specified limit.
- **preprocess_dataframe**: Preprocess the DataFrame by handling missing values, normalizing numeric columns, and converting data types.
- **is_valid_entity_id**: Validate if an entity ID is in the correct format.
- **fetch_direct_parents**: Retrieve the direct parent entities (instances of) an entity from WikiData.
- **determine_most_common_type**: Determine the most common type among given values based on type counts.
- **assign_column_type**: Assign column types based on direct parents or common types.
- **compute_evaluation_metrics**: Compute evaluation metrics (precision, recall, F1 score) for predictions.

### Main Script

1. **Loading and Combining CSV Files**: Load CSV files from the specified directory and combine them into a single DataFrame.
2. **Data Preprocessing**: Preprocess the combined DataFrame.
3. **Assigning Semantic Types**: Assign semantic types to the columns based on direct parents or common types.
4. **Evaluation**: Compute evaluation metrics based on true labels and predicted labels.

### Requirements

- Python 3.8+
- pandas
- requests
- SPARQLWrapper

### Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/KrishnaGupta4135/CTA-and-CTE-for-Semantic-Web-Technologies-using-WikiData.git
    cd your-repository
    ```

2. Install the required packages:
    ```bash
    pip install pandas requests SPARQLWrapper
    ```

3. Run the script:
    ```bash
    python main.py
    ```

## Future Work

Future work could include improving entity resolution techniques, incorporating additional data sources for entity validation, and enhancing the evaluation metrics by considering more advanced metrics.

## Contribution

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
