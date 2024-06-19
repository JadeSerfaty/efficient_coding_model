import os
import json
import boto3
import pandas as pd

DATA_PATH = 'path/to/data'  # Adjust this path
JOB_NAME = 'efficient_coding_job'


def split_data(data_path):
    data = pd.read_csv(os.path.join(data_path, 'rating_data.csv'))
    emotions = data['EMOTION_NAME'].unique()
    durations = data['DURATION'].unique()
    combinations = []

    for emotion in emotions:
        for duration in durations:
            filtered_data = data[(data['EMOTION_NAME'] == emotion) & (data['DURATION'] == duration)]
            if not filtered_data.empty:
                job_name = f"{emotion}_{duration}"
                choice_data_path = os.path.join(data_path, f"{job_name}_choice_data.csv")
                rating_data_path = os.path.join(data_path, f"{job_name}_rating_data.csv")
                filtered_data.to_csv(rating_data_path, index=False)
                combinations.append({
                    "job_name": job_name,
                    "choice_data_path": choice_data_path,
                    "rating_data_path": rating_data_path
                })
    return combinations


def submit_batch_jobs(job_queue, job_definition):
    batch_client = boto3.client('batch')
    combinations = split_data(DATA_PATH)
    array_size = len(combinations)

    response = batch_client.submit_job(
        jobName=JOB_NAME,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        arrayProperties={
            'size': array_size
        },
        containerOverrides={
            'environment': [
                {
                    'name': 'COMBINATIONS',
                    'value': json.dumps(combinations)
                }
            ]
        }
    )
    print(f'Submitted array job - Job ID: {response["jobId"]}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--job_queue', type=str, required=True, help='Name of the AWS Batch job queue')
    parser.add_argument('--job_definition', type=str, required=True, help='Name of the AWS Batch job definition')
    args = parser.parse_args()

    submit_batch_jobs(
        job_queue=args.job_queue,
        job_definition=args.job_definition
    )
