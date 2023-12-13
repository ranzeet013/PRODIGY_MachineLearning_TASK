
# Customer Segmentation with KMeans Clustering

## Objective

The objective of this Python script is to perform customer segmentation using the KMeans clustering algorithm on a dataset of mall customers. Customer segmentation involves grouping similar customers together based on their characteristics, allowing businesses to tailor marketing strategies and services to specific customer segments.

The script aims to achieve the following:

1. **Data Inspection:**
   - Explore the structure and characteristics of the provided 'Mall_Customers.csv' dataset.
   - Convert categorical features, such as 'Gender,' to numerical values for analysis.

2. **Standardizing Dataset:**
   - Standardize the dataset to ensure that all features contribute equally to the clustering algorithm.

3. **Clustering:**
   - Utilize the KMeans clustering algorithm to identify distinct customer groups.
   - Determine the optimal number of clusters using the elbow method.
   - Assign each customer to a specific cluster based on their features.

4. **Visualization:**
   - Provide visualizations, such as the elbow method plot, to aid in understanding the clustering process.

5. **Output:**
   - Save the clustered dataset as 'clustered_dataset.csv' for further analysis and business decision-making.

## Prerequisites

- Python 3
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, yellowbrick

Install the required libraries using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick
```

## Usage

1. **Clone the repository:**

```bash
git clone https://github.com/ranzeet013/PRODIGY_MachineLearning_TASK.git
cd your-repository
```

2. **Run the script:**

```bash
python PRODIGY_ML_02.py
```

3. **View the clustered dataset:**

The clustered dataset will be saved as a CSV file named 'clustered_dataset.csv' in the same directory.

## Script Explanation

1. **Importing Libraries:**
   - Pandas, numpy, matplotlib, and seaborn are imported for data manipulation, numerical operations, and data visualization.

2. **Data Inspection:**
   - The script reads the 'Mall_Customers.csv' dataset and inspects its structure, shape, and statistical information.
   - It maps the 'Gender' column to numerical values.

3. **Standardizing Dataset:**
   - The dataset is standardized using the StandardScaler from scikit-learn.

4. **Clustering:**
   - The script uses the KMeans algorithm to perform clustering.
   - The optimal number of clusters is determined using the elbow method.
   - The dataset is then clustered, and the results are saved to 'clustered_dataset.csv'.

## Visualization

- The elbow method is visualized to determine the optimal number of clusters.

## Output

- The clustered dataset is saved as 'clustered_dataset.csv'.

## Additional Notes

- Consider customizing visualizations for further analysis.
