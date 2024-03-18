import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('/Users/divyakhurana/Downloads/weatherAUS.csv')

# Select numerical features
numerical_features = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 
                      'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 
                      'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
                      'Temp9am', 'Temp3pm']

# Drop rows with missing values in numerical features
df.dropna(subset=numerical_features, inplace=True)

# Encoding 'Location'
le = LabelEncoder()
df['Location_encoded'] = le.fit_transform(df['Location'])

# Including 'Location_encoded' in features
X = df[numerical_features + ['Location_encoded']].values
X = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)

# Create a new dataframe with the principal components and the encoded location
principalDf = pd.DataFrame(data = principalComponents, columns = ['Principal Component 1', 'Principal Component 2'])
principalDf['Location'] = le.inverse_transform(df['Location_encoded'])

# Visualize the first two principal components
# Increased the figure size and DPI for better clarity
plt.figure(figsize=(14, 8), dpi=100)
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', 
                hue='Location', data=principalDf, alpha=0.7, palette='tab20')

# Adjust the legend to spread out vertically; increase the font size if necessary
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, title='Location', title_fontsize='13', fontsize='7')
plt.title('PCA of Rain in Australia Dataset by Location', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=14)
plt.ylabel('Principal Component 2', fontsize=14)

# Adjust the axis labels to ensure they do not overlap
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Optionally, you can save the figure with a higher DPI to ensure it's clear when viewed on different devices
plt.savefig('PCA_by_Location.png', dpi=300, bbox_inches='tight')

plt.show()
