import pickle
import json
import boto3
import os

S3_BUCKET = "efficient-coding-model"


class S3Client:
    def __init__(self):
        self.s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        print("AWS S3 login successful")

    def upload_pickle_to_s3(self, data, key):
        pickle_data = pickle.dumps(data)
        self.s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=pickle_data)

    def upload_json_to_s3(self, data, key):
        self.s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=data)

    def get_json_file_from_s3(self, key):
        response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        json_content = response['Body'].read().decode('utf-8')
        return json.loads(json_content)  # Parse JSON content