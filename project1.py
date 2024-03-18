import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd

## Load the "Rain in Australia" dataset
df_rain_australia = pd.read_csv('/Users/divyakhurana/Downloads/weatherAUS.csv')

## Data manipulation
# Dropping non-numeric columns for PCA
df_rain_australia_numeric = df_rain_australia.select_dtypes(include=[np.number])

## Handling missing values by imputation or dropping
df_rain_australia_numeric.fillna(df_rain_australia_numeric.median(), inplace=True)

## Show content of dataframe
print("Show content of dataframe")
print(df_rain_australia.head())

#######################################################
### STANDARDIZATION OF ATTRIBUTES #####################
#######################################################

X = df_rain_australia_numeric.to_numpy(dtype=np.float32) # Matrix representation of the dataframe

N, M = X.shape # Number of observations and attributes

# Standardize the data
X_tilde = (X - X.mean(axis=0)) / X.std(axis=0)

attribute_names = df_rain_australia_numeric.columns

#######################################################
### MISSING DATA ANALYSIS #############################
#######################################################

print("Count of NA observations (0 means no NA in variable)")
print(df_rain_australia.isnull().sum())

#######################################################
### DESCRIPTIVE SUMMARY STATISTICS ####################
#######################################################

print("Summary statistics")
print(round(df_rain_australia.describe(),2))

#######################################################
### OUTLIER DETECTION #################################
#######################################################

plt.boxplot(X_tilde)
plt.xticks(range(1,M+1), attribute_names, rotation=90)
plt.ylabel('Standardized value')
plt.title('Rain in Australia Data - boxplot')
plt.show()

#######################################################
### CORRELATION MATRIX ################################
#######################################################

print("Correlation Matrix")
print(round(df_rain_australia_numeric.corr(method='pearson'),2))

# Visualizing the Correlation Matrix
plt.figure(figsize=(10,8))
sns.heatmap(df_rain_australia_numeric.corr(), annot=True, fmt=".2f")
plt.title('Correlation Heatmap of Rain in Australia Dataset')
plt.show()

#######################################################
### PCA ANALYSIS ######################################
#######################################################

U, S, Vt = svd(X_tilde, full_matrices=False)
rho = (S**2) / (S**2).sum()

# Plot variance explained
plt.figure(figsize=(5,5),dpi=90)
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
threshold = 0.9
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.show()

# Remember to replace the file path with your actual dataset's location.
