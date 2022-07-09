import pandas as pd
import numpy as np
import os

class Story:
    def __init__(self, user_id, question_id, data) -> None:
        self.user = user_id
        self.question_id = question_id
        self.task_id = data['taskId'][0]
        self.gender = data['gender'][0]
        self.age = data['age'][0]
        self.start_time = data['timestamp'][0]
        self.end_time = data['endtime'][0]
        self.running_time = self.end_time - self.start_time
        self.number_of_steps = len(data['timestamp'])
        self.answer_correct = data['selectedAnswer'][0] == data['correctAnswer'][0]
        self.mental_effort = data['mentalEffort'][0]
        self.experience_level = 'Power BI' if data['powerBIExperience'][0] == 1 else 'Dashboard' if data['dashboardExperience'][0] == 1 else 'None'
        self.confidence = data['confidence'][0]
        self.satisfaction = data['satisfaction'][0]
        self.data = data
        
        for key in [
            'user', 
            'gender', 
            'age',
            'dashboardExperience',
            'powerBIExperience',
            'confidence',
            'satisfaction',
            'questionId',
            'taskId',
            'correctAnswer',
            'mentalEffort',
            'selectedAnswer',
            'endtime'
        ]: self.data.pop(key, None)
        
