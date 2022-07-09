import re
import glob
import csv
from typing import List, Literal
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances


def isCategoricalColumn(colName: str) -> bool: 
    return re.search(r"(?i)\<\bcategor(.*?)\b\>", colName)


def isNumericalColumn(colName: str) -> bool:
    return re.search(r"(?i)\<\bnumer(.*?)\b\>", colName)


# source ./venv/bin/activate


class Stories:
    def __init__(self, question_id: int = None):
        self.data = self.dataframeFromCSV(question_id)
        print(self.data)

    def dataframeFromCSV(self, question_id: int = None):
        csvs = glob.glob('/Users/Thomas/Desktop/Studium/WINF/Masterarbeit/provectories/provectories_python/csv/in/survey/*.csv')
        
        # save cols from first file to keep the same col order through all csvs
        cols_to_use = [str(header) for header in pd.read_csv(csvs[0], sep=';', nrows=0)]
        
        # initialize data dict
        data = {
            col: [] for col in ['feature_vector', 'multiplicity', 'line', 'answer_correct', 'running_time', *cols_to_use]
        }

        keys = [key for key in data.keys() if isCategoricalColumn(key) or isNumericalColumn(key)]

        for key in keys:
            data.pop(key, None)

        for csv in csvs:
            df = pd.read_csv(csv, sep=';', usecols=cols_to_use)[cols_to_use]

            running_time = -1

            for i, row in df.iterrows():
                # skip all question_ids we don't want
                if question_id and question_id != row['questionId']:
                    continue

                if row['triggeredAction'] == 'Root':
                    try:
                        running_time = int(row['endtime']) - int(row['timestamp'])
                    except:
                        print(f"no integer value provided for {row['user']}")
                        running_time = -1

                feature_vector: List[float] = []
                data['feature_vector'].append(feature_vector)
                data['line'].append(f"{row['user']} - {row['questionId']}")
                data['answer_correct'].append(row['selectedAnswer'] == row['correctAnswer'])
                data['running_time'].append(running_time)

                for column in df:
                    cell = row[column]
                    if isCategoricalColumn(column):
                        split_vector = [float(num) for num in cell.split(',')]
                        feature_vector.extend(split_vector)
                        # data[column].append(split_vector)
                    elif isNumericalColumn(column): # TODO: handle numerical cols or leave ou
                        # print(column)
                        # print(sum([float(num) for num in cell.split(',')]))
                        pass
                        # data[column].append(cell)
                    else:
                        data[column].append(cell)

        for i, b in enumerate(data['feature_vector']):
            if (len(b) != 101):
                raise Exception("invalid feature vector length")

        encoded, indicies, counts = np.unique(data['feature_vector'], axis=0, return_inverse=True, return_counts=True)
        data['multiplicity'] = counts[indicies]
        return pd.DataFrame.from_dict(data)


    def calculateDistance(self, metric: Literal['kernel', 'euclidean'] = 'kernel'):
        df = self.data
        X = np.array(df['feature_vector'].values.tolist())

        # for key in df.keys():
        #     if isCategoricalColumn(key):
        #         X = np.array(df[key].values.tolist())
        #         distance = euclidean_distances(X) if metric == 'euclidean' else 1 - rbf_kernel(X)
        #         print(distance)

        distance = euclidean_distances(X) if metric == 'euclidean' else 1 - rbf_kernel(X)

        # raise Exception("hi :)")

        embedded = TSNE(n_components=2, verbose=3, metric='precomputed').fit_transform(distance)

        df["x"] = [row[0] for row in embedded]
        df["y"] = [row[1] for row in embedded]

        # set same coordinate for root state
        coords = None
        
        for i, action in enumerate(df['triggeredAction']):
            if action == 'Root':
                 # get coordinates of first root state
                if not coords:
                    coords = [df.at[i, 'x'], df.at[i, 'y']]

                df.at[i, 'x'] = coords[0]
                df.at[i, 'y'] = coords[1]


    def writeCSV(self, file_name: str, columns_of_interest: List[str] = None):

        keys = ['x', 'y', 'line', 'multiplicity', 'timestamp', *columns_of_interest] if columns_of_interest else self.data.keys()

        with open(
            f'/Users/Thomas/Desktop/Studium/WINF/Masterarbeit/provectories/provectories_python/csv/out/{file_name}.csv', 'w', encoding='UTF8'
        ) as f:
            writer = csv.writer(f)

            csv.writer(f).writerow(keys)

            for i, row in enumerate(self.data['timestamp']):
                writer.writerow([self.data.at[i, key] for key in keys])

            f.close()
