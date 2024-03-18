import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('/Users/divyakhurana/Downloads/weatherAUS.csv')  # Adjust the path to your CSV file

# Selecting the numerical features for PCA
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

# Fill missing values if necessary
df[numerical_features] = df[numerical_features].fillna(df[numerical_features].median())

# Standardize the data
X_std = StandardScaler().fit_transform(df[numerical_features])

# Apply PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_std)

# Get the component loadings (also known as eigenvectors)
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Convert to a DataFrame
loadings_df = pd.DataFrame(loadings, columns=['PC1', 'PC2', 'PC3'], index=numerical_features)

# Plot the loadings for the first three components directly with DataFrame.plot method
ax = loadings_df.plot(kind='bar', figsize=(10, 6))
ax.set_title('PCA Component Loadings')
ax.set_ylabel('Loading Value')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate the x-axis labels for better readability
plt.tight_layout()  # Adjust the layout to make room for the x-axis labels

# Save the figure
plt.savefig('pca_component_loadings.png', dpi=300, bbox_inches='tight')
plt.show()
