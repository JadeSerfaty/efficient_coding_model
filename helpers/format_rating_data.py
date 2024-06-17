import pandas as pd

# Load the CSV file
file_path = 'data/mock/rating_data.csv'
rating_data = pd.read_csv(file_path)

# Create a function to transform the data
def transform_data(df):
    # Transform phase 1 data
    rating_phase1 = df[['videoID', 'emotionName', 'subject_id', 'durationBlackScreen_phase1', 'rating_phase1', 'average_rating', 'variance_rating']].rename(
        columns={
            'videoID': 'VIDEO_ID',
            'emotionName': 'EMOTION_NAME',
            'subject_id': 'SUBJECT_ID',
            'durationBlackScreen_phase1': 'DURATION',
            'rating_phase1': 'RATING',
            'average_rating': 'AVERAGE_RATING',
            'variance_rating': 'VARIANCE_RATING'
        })
    rating_phase1['PHASE'] = 1

    # Transform phase 2 data
    rating_phase2 = df[['videoID', 'emotionName', 'subject_id', 'durationBlackScreen_phase2', 'rating_phase2', 'average_rating', 'variance_rating']].rename(
        columns={
            'videoID': 'VIDEO_ID',
            'emotionName': 'EMOTION_NAME',
            'subject_id': 'SUBJECT_ID',
            'durationBlackScreen_phase2': 'DURATION',
            'rating_phase2': 'RATING',
            'average_rating': 'AVERAGE_RATING',
            'variance_rating': 'VARIANCE_RATING'
        })
    rating_phase2['PHASE'] = 2

    # Combine the two phases
    transformed_data = pd.concat([rating_phase1, rating_phase2], ignore_index=True)

    # Ensure consistent column names in uppercase with underscores
    transformed_data.columns = [col.upper() for col in transformed_data.columns]

    return transformed_data

# Transform the data
transformed_data = transform_data(rating_data)

# Save the transformed data to a new CSV file
transformed_file_path = 'data/mock/rating_data_formatted.csv'
transformed_data.to_csv(transformed_file_path, index=False)

print(f"Transformed data saved to {transformed_file_path}")
