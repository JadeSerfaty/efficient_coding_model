import json
import boto3
from data_formatter import DataFormatter

# Define constants
DATA_PATH = './data/mock/auguste/rating_data_formatted.csv'  # Adjust this path
JOB_NAME = 'MyModelRunnerJob'
JOB_QUEUE = 'MyJobQueue'
JOB_DEFINITION = 'MyJobDefinition'
EMOTIONS = ['joy']  # Example emotions
DURATIONS = [900]  # Example durations
PHASES = [1, 2]  # Example phases

def submit_batch_jobs(data_path=DATA_PATH):
    formatter = DataFormatter(data_path, EMOTIONS, DURATIONS, PHASES)
    combinations = formatter.split_data()
    print(combinations)
    #
    # batch_client = boto3.client('batch')
    # for idx, combination in enumerate(combinations):
    #     response = batch_client.submit_job(
    #         jobName=f"{JOB_NAME}_{combination['job_name']}",
    #         jobQueue=JOB_QUEUE,
    #         jobDefinition=JOB_DEFINITION,
    #         containerOverrides={
    #             'environment': [
    #                 {
    #                     'name': 'CHOICE_DATA',
    #                     'value': combination['choice_data'].to_json()
    #                 },
    #                 {
    #                     'name': 'RATING_DATA',
    #                     'value': combination['rating_data'].to_json()
    #                 },
    #                 {
    #                     'name': 'JOB_NAME',
    #                     'value': combination['job_name']
    #                 }
    #             ]
    #         }
    #     )
    #     print(f'Submitted job {idx + 1}/{len(combinations)} - Job ID: {response["jobId"]}')

if __name__ == "__main__":
    submit_batch_jobs()
