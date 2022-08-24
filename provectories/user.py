import re
from typing import Dict, List
import pandas as pd

from .question import Question

def isCategoricalColumn(colName: str) -> bool: 
    return re.search(r"(?i)\<\bcategor(.*?)\b\>", colName)


def isNumericalColumn(colName: str) -> bool:
    return re.search(r"(?i)\<\bnumer(.*?)\b\>", colName)


class User:
    def __init__(self, df: pd.DataFrame, line):
        self.user_id = df.at[0, 'user']
        self.gender = df.at[0, 'gender']
        self.age = df.at[0, 'age']
        self.dashboard_experience = df.at[0, 'dashboardExperience']
        self.power_bi_experience = df.at[0, 'powerBIExperience']
        self.confidence = df.at[0, 'confidence']
        self.satisfaction = df.at[0, 'satisfaction']
        self.line = line
        self.questions: List[Question] = self._group_df_to_questions(df)
        self.sample = df.at[0, 'user'] == 'sample_solution'

    def __getitem__(self, key):
        return getattr(self, key)

    def _group_df_to_questions(self, df: pd.DataFrame) -> List[Question]:
        data_objects: List[Dict] = []

        for i, row in df.iterrows():
            # skip rows with tracking errors caused by very fast clicking (selection updated faster than filter states)
            if i + 1 < len(df) and (row['timestamp'] + 300) > df.at[(i + 1), 'timestamp']:
                continue

            if row['triggeredAction'] == 'Root':
                # reset data and append it to data_objects
                data = {
                    col: [] for col in [
                        'feature_vector',
                        *[key for key in df.keys() if not isCategoricalColumn(key) and not isNumericalColumn(key)]
                    ]
                }
                data_objects.append(data)
            
            feature_vector: List[float] = []
            data['feature_vector'].append(feature_vector)

            for column in df:
                    cell = row[column]
                    if isCategoricalColumn(column):
                        split_col = [float(num) for num in cell.split(',')]
                        # weight year column more
                        if re.search(r"(?i)\b(.*?)year.year(.*?)\b", column) and 1 in split_col:
                            split_col = [((len(split_col)/2)/len([one for one in split_col if one == 1])) * num for num in split_col]
                        feature_vector.extend(split_col)
                    elif isNumericalColumn(column): # TODO: handle numerical cols or leave out
                        pass
                    else:
                        data[column].append(cell)

            if (len(feature_vector) != 101):
                raise Exception("invalid feature vector length")

        return [Question(pd.DataFrame.from_dict(data)) for data in data_objects]


    def __str__(self):
        return str(self.user_id + " " + str(self.dashboard_experience))

    def get_question_by_id(self, id: int) -> Question:
        for question in self.questions:
            if question.question_id == id:
                return question
         