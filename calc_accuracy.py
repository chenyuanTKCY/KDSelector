import pandas as pd
import os
import argparse

def calc_accuracy(file_path):
    read_path = os.path.join(file_path, "current_accuracy_AUC_PR.csv")
    data = pd.read_csv(read_path)
    columns_to_extract = data.iloc[:, [0, 3]]
    result = columns_to_extract.groupby(columns_to_extract.columns[0])[columns_to_extract.columns[1]].mean().reset_index()
    result.columns = ['Dataset', 'Accuracy']
    output_path = os.path.join(file_path, "cal_accuracy.csv")
    result.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='calc_accuracy',
        description='Script for calculating the accuracy of the training',
    )
    parser.add_argument('-p', '--path', type=str, help='path to the score to save', required=True)
    args = parser.parse_args()
    calc_accuracy(
        file_path=args.path
    )
 