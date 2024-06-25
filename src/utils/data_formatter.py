import pandas as pd

class DataFormatter:
    def __init__(self, data_path, emotions, durations, phases, run_choice=False, run_name="default_run"):
        self.data_path = data_path
        self.emotions = emotions
        self.durations = durations
        self.phases = phases
        self.run_name = run_name
        self.run_choice = run_choice

    def split_data(self):
        data = pd.read_csv(self.data_path)
        combinations = []

        if len(self.durations) > 1:
            # Combinations per subject_id and emotion
            for emotion in self.emotions:
                for subject_id in data['SUBJECT_ID'].unique():
                    filtered_data = data[
                        (data['EMOTION_NAME'].str.lower() == emotion.lower()) &
                        (data['SUBJECT_ID'] == subject_id)
                    ]
                    if not filtered_data.empty:
                        job_name = f"{subject_id}_{emotion}"
                        # choice_data = filtered_data[filtered_data['PHASE'] == self.phases[0]]  # First phase is choice
                        rating_data = filtered_data[filtered_data['PHASE'].isin(self.phases)]  # All phases are rating
                        combinations.append({
                            "job_name": job_name,
                            "choice_data": False, #FIXME choice_data
                            "rating_data": rating_data,
                            "run_name": self.run_name,
                            "run_choice": self.run_choice
                        })
        else:
            # Combinations per subject_id, emotion, and duration
            for emotion in self.emotions:
                for duration in self.durations:
                    for subject_id in data['SUBJECT_ID'].unique():
                        filtered_data = data[
                            (data['EMOTION_NAME'].str.lower() == emotion.lower()) &
                            (data['DURATION'] == duration) &
                            (data['SUBJECT_ID'] == subject_id)
                        ]
                        if not filtered_data.empty:
                            job_name = f"{subject_id}_{emotion}_{duration}"
                            # choice_data = filtered_data[filtered_data['PHASE'] == self.phases[0]]  # First phase is choice
                            rating_data = filtered_data[filtered_data['PHASE'].isin(self.phases)]  # All phases are rating
                            combinations.append({
                                "job_name": job_name,
                                "choice_data": False,
                                "rating_data": rating_data,
                                "run_name": self.run_name,
                                "run_choice": self.run_choice
                            })

        return combinations
