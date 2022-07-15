from typing import List
import pandas as pd

from .state import State

class Question:
    def __init__(self, df: pd.DataFrame):
        self.question_id = df.at[0, 'questionId']
        self.task_id = df.at[0, 'taskId']
        self.correct_answer = df.at[0, 'correctAnswer']
        self.selected_answer = df.at[0, 'selectedAnswer']
        self.answer_correct = self.correct_answer == self.selected_answer
        self.mental_effort = df.at[0, 'mentalEffort']
        self.no_of_steps = df.shape[0]
        self.start_time = df.at[0, 'timestamp']
        self.end_time = df.at[0, 'endtime']
        self.running_time = int(self.end_time - self.start_time) / 1000
        self.states = self._init_states(df)

    def __getitem__(self, key):
        return getattr(self, key)

    def _init_states(self, df: pd.DataFrame) -> List[State]:
        return [State(
            row['timestamp'],
            row['triggeredAction'],
            row['selectedValues'],
            row['filteredValues'],
            row['feature_vector']
        ) for i, row in df.iterrows()]

    def __str__(self):
        return str(str(self.question_id) + " " + str(self.answer_correct))