import json
from src.utils.data_formatter import DataFormatter
import os
import subprocess

# Define constants
DATA_PATH = './data/mock/auguste/rating_data_formatted.csv'  # Adjust this path
EMOTIONS = ['joy', 'anxiety', 'sadness', 'romance', 'disgust']  # Example emotions
DURATIONS = [900, 2600]  # Example durations
PHASES = [1, 2]  # Example phases
RUN_NAME = 'batch_jade'  # Run name to be added for S3 key
RUN_CHOICE = False  # Whether to run the choice model


def prepare_and_run_first_combination():
    formatter = DataFormatter(data_path=DATA_PATH,
                              emotions=EMOTIONS,
                              durations=DURATIONS,
                              phases=PHASES,
                              run_choice=RUN_CHOICE,
                              run_name=RUN_NAME)

    combinations = formatter.split_data()

    # Convert DataFrames to JSON strings
    for combination in combinations:
        combination['choice_data'] = combination['choice_data'].to_json(orient='split') if combination[
            'choice_data'] else False
        combination['rating_data'] = combination['rating_data'].to_json(orient='split')

    # Convert the combinations list to JSON
    combinations_json = json.dumps(combinations)

    # Escape the JSON string to pass it as an environment variable
    escaped_combinations_json = combinations_json.replace('"', '\\"')

    print('Prepared data and saved combinations to combinations.json')

    # Run the Docker container with the first combination
    subprocess.run([
        'docker', 'run', '--rm',
        '-e', f'COMBINATIONS="{escaped_combinations_json}"',
        '-e', 'AWS_BATCH_JOB_ARRAY_INDEX=0',
        '-e', f'AWS_ACCESS_KEY_ID={os.getenv("AWS_ACCESS_KEY_ID")}',
        '-e', f'AWS_SECRET_ACCESS_KEY={os.getenv("AWS_SECRET_ACCESS_KEY")}',
        'efficient_coding_model'
    ])


if __name__ == "__main__":
    prepare_and_run_first_combination()
