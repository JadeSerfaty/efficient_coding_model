import json
import boto3
from src.utils.data_formatter import DataFormatter
import os

# Define constants
DATA_PATH = './data/mock/auguste/rating_data_formatted.csv'  # Adjust this path
JOB_NAME = 'batch_job'
JOB_QUEUE = 'MyJobQueue'
JOB_DEFINITION = 'MyModelRunnerJob'
EMOTIONS = ['joy', 'anxiety']  # Example emotions
DURATIONS = [900, 2600]  # Example durations
PHASES = [1, 2]  # Example phases
RUN_NAME = 'batch_jade'  # Run name to be added for S3 key
RUN_CHOICE = False  # Whether to run the choice model


def submit_batch_jobs():
    formatter = DataFormatter(data_path=DATA_PATH,
                              emotions=EMOTIONS,
                              durations=DURATIONS,
                              phases=PHASES,
                              run_choice=RUN_NAME,
                              run_name=RUN_CHOICE)

    combinations = formatter.split_data()

    # Convert DataFrames to JSON strings
    for combination in combinations:
        combination['choice_data'] = combination['choice_data'].to_json(orient='split')
        combination['rating_data'] = combination['rating_data'].to_json(orient='split')

    # Convert the combinations list to JSON
    combinations_json = json.dumps(combinations)

    batch_client = boto3.client('batch')
    response = batch_client.submit_job(
        jobName=JOB_NAME,
        jobQueue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        arrayProperties={
            'size': len(combinations)
        },
        containerOverrides={
            'environment': [
                {
                    'name': 'COMBINATIONS',
                    'value': combinations_json
                },
                {
                    'name': 'AWS_ACCESS_KEY_ID',
                    'value': os.getenv('AWS_ACCESS_KEY_ID')
                },
                {
                    'name': 'AWS_SECRET_ACCESS_KEY',
                    'value': os.getenv('AWS_SECRET_ACCESS_KEY')
                },
            ]
        }
    )
    print(f'Submitted array job - Job ID: {response["jobId"]}')


if __name__ == "__main__":
    submit_batch_jobs()
