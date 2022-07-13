import re
import glob
import csv
from typing import List, Literal
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

from .user import User

class Provectories:
    def __init__(self):
        self.users = self._data_frame_from_csv()
        self.data = None
        print(self.users)

    def _data_frame_from_csv(self):
        csvs = glob.glob(
            '/Users/Thomas/Desktop/Studium/WINF/Masterarbeit/provectories/provectories_python/csv/in/survey/*.csv'
        )
        
        # save cols from first file to keep the same col order through all csvs
        ordered_cols = [str(header) for header in pd.read_csv(csvs[0], sep=';', nrows=0)]

        return [User(pd.read_csv(csv, sep=';', usecols=ordered_cols)[ordered_cols], idx + 1) for idx, csv in enumerate(csvs)]


    def _calculate_distances(self, data: pd.DataFrame, distance_metric: Literal['kernel', 'euclidean'] = 'kernel') -> pd.DataFrame:
        feature_vectors = data['feature_vector'].values.tolist()
        
        encoded, indicies, counts = np.unique(feature_vectors, axis=0, return_inverse=True, return_counts=True)
        data['multiplicity'] = counts[indicies]

        X = np.array(feature_vectors)
        distance = euclidean_distances(X) if distance_metric == 'euclidean' else 1 - rbf_kernel(X)
        embedded = TSNE(n_components=2, verbose=3, metric='precomputed').fit_transform(distance)

        x = np.array([row[0] for row in embedded])
        y = np.array([row[1] for row in embedded])

        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min)

        y_min = y.min()
        y_max = y.max()
        y = (y - y_min) / (y_max - y_min)

        data['x'] = list(x)
        data['y'] = list(y)

        # get coordinates of first root state
        coords = [data.at[0, 'x'], data.at[0, 'y']]
        
        # unify coordinates for root states
        for i, action in enumerate(data['triggeredAction']):
            if action == 'Root':
                data.at[i, 'x'] = coords[0]
                data.at[i, 'y'] = coords[1]

        return data


    def _write_csv(self, data: pd.DataFrame, file_name: str, columns_of_interest: List[str] = None):
        keys = [
            'x',
            'y',
            'line',
            'multiplicity',
            'timestamp',
            *columns_of_interest
        ] if columns_of_interest else data.keys()

        with open(
            f'/Users/Thomas/Desktop/Studium/WINF/Masterarbeit/provectories/provectories_python/csv/out/{file_name}.csv',
            'w',
            encoding='UTF8'
        ) as f:
            writer = csv.writer(f)

            csv.writer(f).writerow(keys)

            for i, row in enumerate(data['timestamp']):
                writer.writerow([data.at[i, key] for key in keys])

            f.close()

    def create_csv_file(self, file_name: str, question_id: int = None, columns_of_interest: List[str] = None, distance_metric: Literal['kernel', 'euclidean'] = 'kernel'):
        data = self.data[self.data['questionId'] == question_id].reset_index(drop=True) if question_id else self.data
        data = self._calculate_distances(data, distance_metric=distance_metric)
        self._write_csv(data, file_name=file_name, columns_of_interest=columns_of_interest)

