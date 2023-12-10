# Regression Analysis on House Price Dataset

- **Objective**: Predict housing prices based on various features using regression analysis.
- **Dataset Source**: Kaggle, featuring information like bedrooms, square footage, and location.

## Getting Started:

### Prerequisites:

Before running the code, make sure you have the following installed:

- Python (version >= 3.6)
- Jupyter Notebook (optional)

### Installation:

1. Clone the repository:
   ```bash
   git clone https://github.com/ranzeet013/PRODIGY_MachineLearning_TASK.git
   cd your-repository
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage:

1. Open Jupyter Notebook (optional):
   ```bash
   jupyter notebook
   ```
   - Navigate to the `PRODIGY_ML_01.ipynb` notebook.

2. Run the cells in the notebook to execute the regression analysis on the house price dataset.

## Importing Libraries:

- Utilize essential Python libraries for data analysis and visualization:
  - pandas, numpy: Data manipulation and numerical operations.
  - seaborn, matplotlib: Plotting and visualization.
  - yellowbrick: Visualization of machine learning results.
  - scikit-learn: Machine learning tasks (imputation, linear regression).
  - statsmodels: Statistical modeling.

## Data Import and Features:

- Import training and test datasets, concatenate into a single dataframe.

## Data Preprocessing:

- Perform preprocessing steps:
  - Handle missing values using SimpleImputer.
  - Convert categorical variables to numerical codes.
  - Remove columns with excessive missing values.

## SimpleImputer and Data Transformation:

- Use SimpleImputer to handle missing values (strategy: "most_frequent").
- Convert categorical columns to numerical codes.

## Splitting Dataset and Outlier Handling:

- Split dataset into training and testing sets using train_test_split.
- Identify and handle outliers based on the Interquartile Range (IQR).
- Calculate and visualize the IQR for outlier detection.

## Linear Regression and Ordinary Least Squares (OLS):

- Implement Linear Regression using scikit-learn.
- Implement Ordinary Least Squares (OLS) regression using statsmodels.
- Examine coefficients, intercepts, and regression summaries.

## Visualizations and Model Evaluation:

- Create visualizations, including Q-Q plots and residuals plots, using libraries like Yellowbrick and matplotlib.
- Calculate Mean Squared Error (MSE) and R2 Score for model evaluation.

## Uses:

This project serves as a practical demonstration of regression analysis in machine learning, covering key steps from data preprocessing to model evaluation. It can be used as a reference for implementing regression models in real-world scenarios, providing insights into handling missing data, outlier detection, and choosing between scikit-learn and statsmodels for regression tasks.

## Conclusion:

- Comprehensive exploration of regression analysis on a house price dataset.
- Covers data preprocessing, outlier handling, model implementation, and evaluation.
- Utilizes scikit-learn, statsmodels, and visualization libraries for a thorough understanding of the dataset and predictive models.
- Includes Interquartile Range (IQR) for outlier detection.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.






