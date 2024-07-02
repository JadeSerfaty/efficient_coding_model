import json
import boto3
from src.utils.data_formatter import DataFormatter
import os
from src.utils.s3_client import S3Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define constants
DATA_PATH = './data/main_study/v2/rating_data_formatted.csv'  # Adjust this path
JOB_NAME = 'main_study_v2_training_separate_sigmas_new_model'
JOB_QUEUE = 'MyJobQueue'
JOB_DEFINITION = 'MyModelRunnerJob'
EMOTIONS = ['joy', 'sadness', 'romance', 'disgust', 'anxiety']  # Example emotions
DURATIONS = [900, 2600]  # Example durations #2600
PHASES = [1, 2]  # Example phases
RUN_NAME = 'main_study_v2_training_separate_sigmas_new_model'  # Run name to be added for S3 key
RUN_CHOICE = False  # Whether to run the choice model
COMBINATIONS_KEY_S3 = f"{RUN_NAME}/combinations.json"

def submit_batch_jobs():
    formatter = DataFormatter(data_path=DATA_PATH,
                              emotions=EMOTIONS,
                              durations=DURATIONS,
                              phases=PHASES,
                              run_choice=RUN_CHOICE,
                              run_name=RUN_NAME)

    combinations = formatter.split_data()

    # Convert DataFrames to JSON strings
    for combination in combinations:
        combination['choice_data'] = combination['choice_data'].to_json(orient='split') if combination['choice_data'] else None
        combination['rating_data'] = combination['rating_data'].to_json(orient='split')

    print('Prepared data and saved combinations to combinations.json')
    # Convert the combinations list to JSON
    s3_client = S3Client()
    combinations_json = json.dumps(combinations)
    s3_client.upload_json_to_s3(combinations_json, COMBINATIONS_KEY_S3)
    print('Uploaded combinations to S3')

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
                    'name': 'COMBINATIONS_KEY_S3',
                    'value': COMBINATIONS_KEY_S3
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
