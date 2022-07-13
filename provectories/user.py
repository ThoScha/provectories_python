import re
from typing import List
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

    def _group_df_to_questions(self, df: pd.DataFrame) -> List[Question]:
        questions: List[Question] = []

        for i, row in df.iterrows():
            if row['triggeredAction'] == 'Root':
                # create new question from previously aggregated data
                if i > 0:
                    questions.append(Question(pd.DataFrame.from_dict(data)))
                # reset data for next question
                data = {
                    col: [] for col in [
                        'feature_vector',
                        *[key for key in df.keys() if not isCategoricalColumn(key) and not isNumericalColumn(key)]
                    ]
                }
            
            feature_vector: List[float] = []
            data['feature_vector'].append(feature_vector)

            for column in df:
                    cell = row[column]
                    if isCategoricalColumn(column):
                        split_col = [float(num) for num in cell.split(',')]
                        # weight year column more
                        if "year.year" in column:
                            split_col = [num * 5 for num in split_col]
                        feature_vector.extend(split_col)
                    elif isNumericalColumn(column): # TODO: handle numerical cols or leave out
                        pass
                    else:
                        data[column].append(cell)

            if (len(feature_vector) != 101):
                raise Exception("invalid feature vector length")

        return questions

    def __str__(self):
        return str(self.user_id + " " + str(self.dashboard_experience))
