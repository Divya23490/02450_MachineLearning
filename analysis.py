import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load the dataset
df = pd.read_csv('/Users/divyakhurana/Downloads/weatherAUS.csv')

# Preprocess the data: convert 'RainTomorrow' to numerical values, handle NaNs
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Drop rows with NaN values in 'Humidity3pm' or 'RainTomorrow'
df = df.dropna(subset=['Humidity3pm', 'RainTomorrow'])

# Selecting two attributes
x = df[['Humidity3pm']]  # Predictor, double square brackets to keep the dataframe structure
y = df['RainTomorrow']   # Response

# Fit the logistic regression model
model = LogisticRegression()
model.fit(x, y)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.5)

# Generate a sequence of values for humidity from min to max
x_test = np.linspace(df['Humidity3pm'].min(), df['Humidity3pm'].max(), 300).reshape(-1, 1)

# Predict probabilities for the test sequence
y_prob = model.predict_proba(x_test)[:, 1]

# Plot the logistic curve
plt.plot(x_test, y_prob, color='orange', linewidth=2)

# Labels and title
plt.xlabel('Humidity at 3 PM')
plt.ylabel('Probability of Rain Next Day')
plt.title('Logistic Regression: Humidity3pm vs Probability of RainTomorrow')

plt.show()
