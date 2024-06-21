import json
import os
import pandas as pd
from src.utils.data_formatter import DataFormatter

# Define constants for local testing
DATA_PATH = './data/mock/auguste/rating_data_formatted.csv'
RUN_NAME = 'test'
EMOTIONS = ['joy', 'anxiety']
DURATIONS = [900, 2600]
PHASES = [1, 2]

def generate_test_data():
    # Create mock data directory
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Generate sample data
    data = {
        "VIDEO_ID": [1191, 1281, 1848, 1492, 1777, 730, 1698, 83, 46, 190, 844],
        "EMOTION_NAME": ['joy', 'joy', 'joy', 'anxiety', 'anxiety', 'anxiety', 'joy', 'joy', 'anxiety', 'anxiety', 'joy'],
        "SUBJECT_ID": ['4G1tK8qkj0PHUfED8hXzqbo49BC2'] * 11,
        "DURATION": [2600, 900, 900, 2600, 2600, 900, 900, 2600, 900, 2600, 900],
        "RATING": [4.82, 1.1, 2.21, 6.49, 2.95, 4.82, 3.51, 5.69, 3.0, 4.08, 1.92],
        "AVERAGE_RATING": [4.265, 1.835, 1.99, 5.81, 3.255, 4.955, 3.335, 5.725, 2.99, 4.37, 2.08],
        "VARIANCE_RATING": [0.784, 1.039, 0.311, 0.961, 0.431, 0.190, 0.247, 0.049, 0.014, 0.410, 0.226],
        "PHASE": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # Adjust PHASE as needed
    }

    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)

def run_local_test():
    # Generate test data
    generate_test_data()

    # Split the data and prepare for submission
    formatter = DataFormatter(DATA_PATH, EMOTIONS, DURATIONS, PHASES, RUN_NAME)
    combinations = formatter.split_data()

    # Convert DataFrames to JSON strings
    for combination in combinations:
        combination['choice_data'] = combination['choice_data'].to_json(orient='split')
        combination['rating_data'] = combination['rating_data'].to_json(orient='split')

    combinations_json = json.dumps(combinations)

    # Simulate array job by running the Docker container for each combination
    for idx, combination in enumerate(combinations):
        os.system(
            f'docker run --rm -e COMBINATIONS=\'{combinations_json}\' '
            f'-e AWS_BATCH_JOB_ARRAY_INDEX={idx} '
            f'-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID '
            f'-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY '
            f'637423572282.dkr.ecr.eu-west-2.amazonaws.com/efficient_coding_model:latest'
        )


if __name__ == "__main__":
    run_local_test()
