import matplotlib.pyplot as plt
import pandas as pd

# Load your dataset
df = pd.read_csv('/Users/divyakhurana/Downloads/weatherAUS.csv')

# List of numerical attributes to analyze
numerical_attributes = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am',
                        'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm',
                        'Temp9am', 'Temp3pm']

# Loop through each numerical attribute
for attribute in numerical_attributes:
    # Generate histogram for the attribute
    plt.figure(figsize=(10, 6))
    df[attribute].hist(bins=30)
    plt.title(f'Histogram of {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Frequency')
    plt.savefig(f'{attribute}_histogram.png')  # Save the histogram
    plt.show()

    # Print summary statistics for the attribute
    print(f'Summary statistics for {attribute}:')
    print(df[attribute].describe())
    print('\n' + '-'*50 + '\n')
