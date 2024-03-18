import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('/Users/divyakhurana/Downloads/weatherAUS.csv')

# Selecting the numerical features
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
df_numerical = df[numerical_features].dropna()

# Standardize the data
X_std = StandardScaler().fit_transform(df_numerical)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_std)

# Variance explained by each component
explained_variance = pca.explained_variance_ratio_

# Creating a scree plot
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance)
plt.xlabel('PCA Components')
plt.ylabel('Variance Explained')
plt.title('Scree Plot for PCA Components')
plt.show()


# Assuming PCA has already been applied as in Task 1

# Get the absolute value of component loadings
loadings = pd.DataFrame(np.abs(pca.components_), columns=df_numerical.columns)

# Plotting the loadings for the first two principal components
loadings = loadings.head(2).T
loadings.columns = ['PC1', 'PC2']
loadings.plot(kind='bar', figsize=(10, 6))
plt.title('Absolute Loadings for First Two PCA Components')
plt.ylabel('Loadings Value')
plt.show()

# Assuming PCA has already been applied as in Task 1

# Take the first two principal components for visualization
pca_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])

# Assuming 'RainTomorrow' has been converted to numerical values
df_copy = df.copy()

# Convert 'RainTomorrow' to numeric
df_copy['RainTomorrow'] = df_copy['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Now drop rows with any NaNs (including 'RainTomorrow')
df_numerical = df_copy[numerical_features + ['RainTomorrow']].dropna()

# Separate the features and the target
X = df_numerical[numerical_features]
y = df_numerical['RainTomorrow']

# Standardize the features
X_std = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # You can change the number of components
X_pca = pca.fit_transform(X_std)

# Combine PCA results with 'RainTomorrow' for the scatter plot
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['RainTomorrow'] = y.values  # This ensures the lengths match

# Scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['RainTomorrow'], alpha=0.5)
plt.title('PCA of Rain in Australia Dataset by Rain Tomorrow')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter)
plt.show()

