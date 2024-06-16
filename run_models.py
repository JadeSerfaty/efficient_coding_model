import argparse
from models import ModelRunner


def main(emotion="anxiety", duration="long", run_choice=False, use_mock_data=False):
    model_runner = ModelRunner(emotion=emotion, duration=duration, use_mock_data=use_mock_data, run_choice=run_choice)
    model_runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_choice", action="store_true",
                        help="Run the choice model after the efficient coding model")
    parser.add_argument("--use_mock_data", action="store_true", help="Use mock data instead of real data")
    parser.add_argument("--emotion", type=str, default="anxiety", help="Emotion to filter the data")
    parser.add_argument("--duration", type=str, choices=["short", "long", "both"], default="long",
                        help="Duration to filter the data")
    args = parser.parse_args()
    main(emotion=args.emotion, duration=args.duration, run_choice=args.run_choice, use_mock_data=args.use_mock_data)
