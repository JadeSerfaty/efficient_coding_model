import pandas as pd
import os
import pickle
import sys

def compute_difference(row):
    return row['rating_phase2'] - row['rating_phase1']

def main(index):
    # Load the mock data
    data_path = '/data/mock/rating_data.csv'
    data = pd.read_csv(data_path)

    # Create a directory to save results
    results_dir = 'data/mock/results'
    os.makedirs(results_dir, exist_ok=True)

    # Process the specified row
    row = data.iloc[int(index)]
    result = compute_difference(row)
    result_path = os.path.join(results_dir, f'result_{index}.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'Saved result for row {index} to {result_path}')

if __name__ == "__main__":
    index = sys.argv[1]
    main(index)
