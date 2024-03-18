import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np
import statsmodels.api as sm

# Load the dataset
df = pd.read_csv('/Users/divyakhurana/Downloads/weatherAUS.csv')

# Preprocess the data
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})
df.dropna(subset=['Humidity3pm', 'RainTomorrow'], inplace=True)

# Define predictors and response
X = df[['Humidity3pm']]  # Predictor
y = df['RainTomorrow']   # Response

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Calculate the AUC-ROC
auc_roc = roc_auc_score(y_test, probabilities)

# To calculate a pseudo R-squared, we can use statsmodels


# Add a constant to the predictor variable set to represent the intercept
X_train_with_const = sm.add_constant(X_train)

# Fit the logistic regression model
logit_model = sm.Logit(y_train, X_train_with_const)
fitted_model = logit_model.fit()

# Calculate the pseudo R-squared
pseudo_r_squared = fitted_model.prsquared

print(f'AUC-ROC: {auc_roc}')
print(f'Pseudo R-squared: {pseudo_r_squared}')
