import subprocess

# List of durations and emotions to loop through
durations = ["long", "short"]
emotions = ["JOY", "ANXIETY"]

# Loop through each combination of duration and emotion
for duration in durations:
    for emotion in emotions:
        # Build the command to run the model_runner.py script
        command = [
            "python3",
            "src/model_runner.py",
            "--use_mock_data",
            f"--duration={duration}",
            f"--emotion={emotion.upper()}"
        ]

        # Print the command for debugging purposes
        print(f"Running command: {' '.join(command)}")

        # Run the command
        subprocess.run(command)
