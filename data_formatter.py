import pandas as pd

class DataFormatter:
    def __init__(self, data_path, emotions, durations, phases):
        self.data_path = data_path
        self.emotions = emotions
        self.durations = durations
        self.phases = phases

    def split_data(self):
        data = pd.read_csv(self.data_path)
        combinations = []

        for emotion in self.emotions:
            for duration in self.durations:
                filtered_data = data[(data['EMOTION_NAME'].str.lower() == emotion.lower()) & (data['DURATION'] == duration)]
                if not filtered_data.empty:
                    job_name = f"{emotion}_{duration}"
                    choice_data = filtered_data[filtered_data['PHASE'] == self.phases[0]]  # First phase is choice
                    rating_data = filtered_data[filtered_data['PHASE'].isin(self.phases)]  # All phases are rating
                    combinations.append({
                        "job_name": job_name,
                        "choice_data": choice_data,
                        "rating_data": rating_data
                    })
        return combinations
