
# Data Analysis Report

## Dataset Overview

The dataset contains **10000 rows** and **23 columns**. Here's the data type breakdown:
- **int64**: 13 columns
- **object**: 7 columns
- **float64**: 3 columns

- **5 columns** have missing values.
- **18 columns** are complete.

## Key Insights

### Numerical Features
- **work_text_reviews_count**: Mean = 2919.96, Std Dev = 6124.38
- **ratings_1**: Mean = 1345.04, Std Dev = 6635.63
- **ratings_2**: Mean = 3110.89, Std Dev = 9717.12
- **ratings_3**: Mean = 11475.89, Std Dev = 28546.45
- **ratings_4**: Mean = 19965.70, Std Dev = 51447.36
- **ratings_5**: Mean = 23789.81, Std Dev = 79768.89

### Categorical Features
- **isbn**: Top categories - 1, 1, 1, 1, 1

## Visualizations

![work_text_reviews_count_plot.png](goodreads\work_text_reviews_count_plot.png)
![ratings_1_plot.png](goodreads\ratings_1_plot.png)
![authors_categories.png](goodreads\authors_categories.png)

## Recommendations

1. Address missing data using imputation or removal.
2. Investigate relationships between variables.
3. Explore advanced techniques for deeper insights.

---

*Generated dynamically using Python.*
