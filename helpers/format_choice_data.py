import pandas as pd
import re

# Load the choice data CSV file
choice_data_path = '../choice_data_test.csv'
choice_data = pd.read_csv(choice_data_path)

# Function to convert column names to uppercase with underscores
def convert_column_name(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).upper()

# Rename the columns
choice_data.columns = [convert_column_name(col) for col in choice_data.columns]

# Save the transformed choice data to a new CSV file
transformed_choice_data_path = '../choice_data_test_formatted.csv'
choice_data.to_csv(transformed_choice_data_path, index=False)

print(f"Transformed choice data saved to {transformed_choice_data_path}")
