import glob
import csv
from typing import List
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

# source ./venv/bin/activate

def dataframeFromCSV():
    data = {
        'timestamp': [],
        'line': [],
        'user': [],
        'triggeredAction': [],
        'selectedValues': [],
        'filteredValues': [],
        'feature_vector': [],
        'x': [],
        'y': [],
        'multiplicity': []
    }

    csvs = glob.glob('/Users/Thomas/Desktop/Studium/WINF/Masterarbeit/provectories/provectories_python/csv/in/*.csv')
    
    # save cols from first file to keep the same col order through all csvs
    cols_to_use = [str(header) for header in pd.read_csv(csvs[0], sep=';', nrows=0)]

    for i, csv in enumerate(csvs):
        df = pd.read_csv(csv, sep=';', usecols=cols_to_use)[cols_to_use]
        # df.reset_index()
        for index, row in df.iterrows():
            feature_vector: List[float] = []
            data['timestamp'].append(row[0])
            data['user'].append(row[1])
            data['triggeredAction'].append(row[2])
            data['selectedValues'].append(row[3])
            data['filteredValues'].append(row[4])
            data['line'].append(i + 1)
            data['feature_vector'].append(feature_vector)
            # fill feature_vector
            for idx, column in enumerate(df):
                if idx < 5:  # skip timestamp, user, action
                    continue
                else:
                    cell = []
                    print(row[column])
                    for num in row[column].split(','):
                        cell.append(float(num))
                    if "[categorical]" in column:  # categorical values encoded
                        feature_vector.extend(cell)
                    else:  # TODO: numerical values
                        pass

    
    encoded, indicies, counts = np.unique(data['feature_vector'], axis=0, return_inverse=True, return_counts=True)
    data['multiplicity'] = counts[indicies]

    return data

df = dataframeFromCSV()

X = np.array(df["feature_vector"])

test = 1 - rbf_kernel(X)
# 1 - everything
# test = euclidean_distances(X)
print(test)

X_embedded = TSNE(n_components=2, verbose=3, metric='precomputed').fit_transform(test)

# open_embedded = openTSNE.TSNE(perplexity=30, n_jobs=8, random_state=42, verbose=True).fit(X)
# print(open_embedded)

for row in X_embedded:
    df["x"].append(row[0])
    df["y"].append(row[1])

for i, action in enumerate(df['triggeredAction']):
    if action == 'Root':
        print(df['x'][i])
        print(df['y'][i])
        print(df['multiplicity'][i])

with open('/Users/Thomas/Desktop/Studium/WINF/Masterarbeit/provectories/provectories_python/csv/out/test.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(['timestamp', 'x', 'y', 'line', 'user', 'triggeredAction', 'selectedValues', 'filteredValues', f"multiplicity[{min(df['multiplicity'])};{max(df['multiplicity'])}]"])
    for i, row in enumerate(df['timestamp']):
        writer.writerow([
            df['timestamp'][i],
            df['x'][i],
            df['y'][i],
            df['line'][i],
            df['user'][i],
            df['triggeredAction'][i],
            df['selectedValues'][i],
            df['filteredValues'][i],
            df['multiplicity'][i]
        ])

    f.close()
