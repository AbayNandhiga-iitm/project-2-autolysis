# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "numpy"
# ]
# ///

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def is_meaningful_column(column, df, unique_min=0.1, unique_max=0.9):
    """Determine if a column is meaningful for visualization."""
    unique_ratio = df[column].nunique() / len(df[column])
    return unique_min < unique_ratio < unique_max  # Exclude ID-like or overly sparse/dense columns

def analyze_categorical_column(column, df):
    """Generate insights for categorical columns."""
    value_counts = df[column].value_counts().head(5)  # Top 5 categories
    return value_counts

def generate_plots(df, output_dir):
    """Generate a variety of plots based on the dataset."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_paths = []

    # Safely identify numerical columns that are meaningful for plotting
    numerical_columns = [col for col in df.select_dtypes(include=[np.number]).columns if is_meaningful_column(col, df)]
    if len(numerical_columns) == 0:
        print("No meaningful numerical columns available for plotting.")
    else:
        # 1. Histogram for the first meaningful numerical column
        try:
            column = numerical_columns[0]
            plt.figure(figsize=(8, 6))
            sns.histplot(df[column], kde=True, color='teal', bins=30)
            plt.title(f'Histogram of {column}', fontsize=14)
            plt.xlabel(column, fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plot_path = os.path.join(output_dir, f'{column}_histogram.png')
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
        except Exception as e:
            print(f"Error plotting histogram for column {column}: {e}")

        # 2. Box Plot for the second meaningful numerical column (if available)
        if len(numerical_columns) > 1:
            try:
                column = numerical_columns[1]
                plt.figure(figsize=(8, 6))
                sns.boxplot(y=df[column], color='orange')
                plt.title(f'Box Plot of {column}', fontsize=14)
                plt.ylabel(column, fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plot_path = os.path.join(output_dir, f'{column}_box_plot.png')
                plt.savefig(plot_path)
                plot_paths.append(plot_path)
                plt.close()
            except Exception as e:
                print(f"Error plotting box plot for column {column}: {e}")

    # Generate plots for categorical columns (limit to 1 plot)
    categorical_columns = [col for col in df.select_dtypes(exclude=[np.number]).columns if is_meaningful_column(col, df)]
    for column in categorical_columns[:1]:  # Limit to 1 categorical column
        try:
            plt.figure(figsize=(8, 6))
            sns.countplot(
                data=df, 
                y=column, 
                hue=column,  # Set the hue to the y variable
                order=df[column].value_counts().index[:5], 
                palette='viridis',
                legend=False  # Disable legend as hue is being used
            )
            plt.title(f'Top Categories in {column}', fontsize=14)
            plt.ylabel(column, fontsize=12)
            plt.xlabel("Count", fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            plot_path = os.path.join(output_dir, f'{column}_categories.png')
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
        except Exception as e:
            print(f"Error plotting categories for column {column}: {e}")

    return plot_paths[:2]  # Limit to a maximum of 2 plots

def generate_readme(output_dir, df, plot_paths):
    """Generate a detailed and engaging README file summarizing the analysis."""

    # Data type summary
    column_types = df.dtypes.value_counts()
    dtype_summary = "\n".join([f"- **{dtype}**: {count} columns" for dtype, count in column_types.items()])

    # Extract dataset details
    num_rows = df.shape[0]
    num_columns = df.shape[1]
    missing_count = df.isnull().sum().sum()
    missing_columns = df.isnull().any().sum()
    non_missing_columns = num_columns - missing_columns

    # Meaningful insights for numerical columns (excluding IDs, etc.)
    meaningful_numerical_columns = [col for col in df.select_dtypes(include=[np.number]).columns if is_meaningful_column(col, df)]
    numerical_summary = "\n".join(
        [f"- **{col}**: Mean = {df[col].mean():.2f}, Std Dev = {df[col].std():.2f}" for col in meaningful_numerical_columns if df[col].nunique() > 2]
    )  # Only include columns with more than 2 unique values for meaningful stats

    # Insights for categorical columns
    categorical_summary = "\n".join(
        [f"- **{col}**: Top 5 categories - {', '.join(map(str, analyze_categorical_column(col, df)))}" for col in df.select_dtypes(exclude=[np.number]).columns[:1]]
    )

    # Generate the visualizations markdown
    plots_markdown = "\n".join(
        [f"![{os.path.basename(plot)}]({os.path.join(output_dir, os.path.basename(plot))})" for plot in plot_paths]
    ) if plot_paths else "No visualizations were generated due to data constraints or processing errors."

    # Constructing the readme content
    readme_content = f"""
# Data Analysis Report

Welcome to the data analysis report for your dataset! This document highlights the key findings and provides visualizations that help understand the data better. Dive in to explore the insights revealed during our analysis.

---

## Dataset Overview

The dataset consists of **{num_rows} rows** and **{num_columns} columns**, offering a diverse range of information to analyze. Here's a breakdown of the column data types:
{dtype_summary}

Of these:
- **{missing_columns} columns** contain missing values.
- **{non_missing_columns} columns** are fully populated.

---

## Key Insights

### Numerical Features:
{numerical_summary}

### Categorical Features:
{categorical_summary}

---

## Visualizations

To complement the analysis, the following visualizations were created:

{plots_markdown}

---

## Next Steps

Based on this initial exploration, consider the following recommendations for further analysis:
1. Investigate missing data and decide on appropriate strategies (e.g., imputation or removal).
2. Explore the relationships between key variables using advanced statistical models or machine learning techniques.
3. Consider deeper analysis on outliers, anomalies, or trends observed in the visualizations.

Thank you for reviewing this report! We hope these insights help you make informed decisions or further explore the dataset. Stay curious and keep exploring!

---
*Generated dynamically using Python. Have a great day analyzing your data!*
"""

    # Save the generated README content to a file
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"README.md created successfully: {readme_path}")

def main(file_path):
    """Main function to run the analysis."""
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print(f"Dataset loaded: {file_path}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    # Determine output directory based on input file name
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = base_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        plot_paths = generate_plots(df, output_dir)
        generate_readme(output_dir, df, plot_paths)
        print("Analysis complete!")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <path_to_csv_file>")
    else:
        main(sys.argv[1])
