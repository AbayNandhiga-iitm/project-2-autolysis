# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "os",
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
    return unique_min < unique_ratio < unique_max

def analyze_categorical_column(column, df, top_n=5):
    """Generate insights for categorical columns."""
    return df[column].value_counts().head(top_n)

def generate_plots(df, output_dir):
    """Generate a variety of plots based on the dataset."""
    os.makedirs(output_dir, exist_ok=True)

    plot_paths = []

    # Numerical columns
    numerical_columns = [col for col in df.select_dtypes(include=[np.number]).columns if is_meaningful_column(col, df)]
    for i, column in enumerate(numerical_columns[:2]):  # Limit to first 2 meaningful columns
        try:
            plt.figure(figsize=(8, 6))
            if i == 0:  # Histogram for the first column
                sns.histplot(df[column], kde=True, color='teal', bins=30)
                plt.title(f'Histogram of {column}')
            else:  # Box plot for the second column
                sns.boxplot(y=df[column], color='orange')
                plt.title(f'Box Plot of {column}')
            
            plt.xlabel(column)
            plt.ylabel("Frequency" if i == 0 else column)
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plot_path = os.path.join(output_dir, f'{column}_plot.png')
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
        except Exception as e:
            print(f"Error plotting {column}: {e}")

    # Categorical columns
    categorical_columns = [col for col in df.select_dtypes(exclude=[np.number]).columns if is_meaningful_column(col, df)]
    for column in categorical_columns[:1]:  # Limit to first categorical column
        try:
            plt.figure(figsize=(8, 6))
            sns.countplot(
                data=df, 
                y=column, 
                order=df[column].value_counts().index[:5],
                palette='viridis'
            )
            plt.title(f'Top Categories in {column}')
            plt.ylabel(column)
            plt.xlabel("Count")
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            plot_path = os.path.join(output_dir, f'{column}_categories.png')
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plt.close()
        except Exception as e:
            print(f"Error plotting categories for column {column}: {e}")

    return plot_paths

def generate_readme(output_dir, df, plot_paths):
    """Generate a detailed README file summarizing the analysis."""
    column_types = df.dtypes.value_counts()
    dtype_summary = "\n".join([f"- **{dtype}**: {count} columns" for dtype, count in column_types.items()])

    num_rows, num_columns = df.shape
    missing_count = df.isnull().sum().sum()
    missing_columns = df.isnull().any().sum()
    non_missing_columns = num_columns - missing_columns

    numerical_summary = "\n".join(
        [
            f"- **{col}**: Mean = {df[col].mean():.2f}, Std Dev = {df[col].std():.2f}"
            for col in df.select_dtypes(include=[np.number]).columns
            if is_meaningful_column(col, df)
        ]
    )

    categorical_summary = "\n".join(
        [
            f"- **{col}**: Top categories - {', '.join(map(str, analyze_categorical_column(col, df)))}"
            for col in df.select_dtypes(exclude=[np.number]).columns[:1]
        ]
    )

    plots_markdown = "\n".join(
        [f"![{os.path.basename(plot)}]({plot})" for plot in plot_paths]
    ) if plot_paths else "No visualizations were generated."

    readme_content = f"""
# Data Analysis Report

## Dataset Overview

The dataset contains **{num_rows} rows** and **{num_columns} columns**. Here's the data type breakdown:
{dtype_summary}

- **{missing_columns} columns** have missing values.
- **{non_missing_columns} columns** are complete.

## Key Insights

### Numerical Features
{numerical_summary}

### Categorical Features
{categorical_summary}

## Visualizations

{plots_markdown}

## Recommendations

1. Address missing data using imputation or removal.
2. Investigate relationships between variables.
3. Explore advanced techniques for deeper insights.

---

*Generated dynamically using Python.*
"""

    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"README.md created successfully at {readme_path}")

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
