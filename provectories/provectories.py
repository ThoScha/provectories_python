import glob
import csv
from typing import Dict, List, Literal, OrderedDict
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

from .user import User

class Provectories:
    def __init__(self):
        self.users = self._data_frame_from_csv()

    def _data_frame_from_csv(self):
        csvs = glob.glob(
            '/Users/Thomas/Desktop/Studium/WINF/Masterarbeit/provectories/provectories_python/csv/in/survey/*.csv'
        )
        # save cols from first file to keep the same col order through all csvs
        ordered_cols = [str(header) for header in pd.read_csv(csvs[0], sep=';', nrows=0)]

        return [User(pd.read_csv(csv, sep=';', usecols=ordered_cols)[ordered_cols], idx + 1) for idx, csv in enumerate(csvs)]

    def _aggregate_df_from_users(self, question_id: int, user_cols: List[str] = [], quest_cols: List[str] = []) -> pd.DataFrame:
        user_cols.extend(['line', 'user_id'])
        quest_cols.append('question_id')
        state_cols = ['timestamp', 'feature_vector', 'selected_values', 'filtered_values', 'triggered_action']

        data: Dict[str, List] = {col: [] for col in [*user_cols, *quest_cols, *state_cols]}

        for user in self.users:
            question = user.get_question_by_id(question_id)
            for state in question.states:
                for col in user_cols:
                    data[col].append(user[col])
                for col in quest_cols:
                    data[col].append(question[col])
                for col in state_cols:
                    data[col].append(state[col])
        
        return pd.DataFrame.from_dict(data)

    def _calculate_distances(self, data: pd.DataFrame, distance_metric: Literal['kernel', 'euclidean'] = 'kernel') -> pd.DataFrame:
        feature_vectors = data['feature_vector'].values.tolist()

        encoded, indicies, counts = np.unique(feature_vectors, axis=0, return_inverse=True, return_counts=True)
        data['multiplicity'] = counts[indicies]

        X = np.array(feature_vectors)
        distance = euclidean_distances(X) if distance_metric == 'euclidean' else 1 - rbf_kernel(X)
        embedded = TSNE(n_components=2, verbose=3, metric='precomputed').fit_transform(distance)

        data['x'] = embedded[:,0]
        data['y'] = embedded[:,1]

        # get coordinates of first root state
        coords = [data.at[0, 'x'], data.at[0, 'y']]
        
        # unify coordinates for root states
        for i, action in enumerate(data['triggered_action']):
            if action == 'Root':
                data.at[i, 'x'] = coords[0]
                data.at[i, 'y'] = coords[1]
        
        return data.drop(columns='feature_vector')

    def _write_csv(self, data: pd.DataFrame, file_name: str):
        keys = data.keys()

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

    def create_csv_file(
        self,
        file_name: str,
        question_id: int,
        user_cols: List[str] = [],
        quest_cols: List[str] = [],
        distance_metric: Literal['kernel', 'euclidean'] = 'kernel'
    ):
        data = self._aggregate_df_from_users(question_id, user_cols, quest_cols)
        data = self._calculate_distances(data, distance_metric)
        self._write_csv(data, file_name)
        print('csv file is ready')

    def evaluate_data_sets(self):
        def print_seperator():
            print("__________________________________________________")
            print()
        
        def print_list_item_overview(input_list: List[int or str]):
            keys = list(OrderedDict.fromkeys(input_list).keys())
            keys.sort()
            for key in keys:
                print(f"{key}: ", str(len([item for item in input_list if item == key])))
        
        users = self.users
        total_no_of_users = len(users)

        user_cols = [
            'user_id',
            'gender',
            'age',
            'dashboard_experience',
            'power_bi_experience',
            'confidence',
            'satisfaction'
        ]

        question_cols = [
            'question_id',
            'task_id',
            'correct_answer',
            'selected_answer',
            'answer_correct',
            'mental_effort',
            'no_of_steps',
            'start_time',
            'end_time',
            'running_time'
        ]

        user_data: Dict[str, List[Any]] = {col: [] for col in user_cols}
        question_data: Dict[str, List[Any]] = {col: [] for col in [*user_cols, *question_cols]}

        for user in users:
            for col in user_cols:
                user_data[col].append(user[col])
            for question in user.questions:
                for col in user_cols:
                    question_data[col].append(user[col])
                for col in question_cols:
                    question_data[col].append(question[col])

        user_df = pd.DataFrame.from_dict(user_data)
        question_df = pd.DataFrame.from_dict(question_data)

        ###########################
        ##  User Evaluation  ##
        ###########################

        print_seperator()
        print("USER EVALUATION")
        print_seperator()

        # user distribution
        print("Total no of users: ", total_no_of_users)
        print_seperator()

 # Avg result per dashboard experience
        print("Average correctness per dashboard experience")
        print()
        df_agg = question_df.groupby(['dashboard_experience', 'user_id', 'answer_correct'])['answer_correct'].count().reset_index(name='count')
        avg_res_dash_exp = df_agg.groupby(['dashboard_experience', 'answer_correct'])['count'].mean().reset_index(name='avg')
        avg_res_dash_exp = avg_res_dash_exp[avg_res_dash_exp['answer_correct']].reset_index(drop=True)
        user_count_dash_exp = question_df.groupby(['dashboard_experience'])['user_id'].nunique().reset_index(name='user_count')
        avg_res_dash_exp = pd.merge(avg_res_dash_exp, user_count_dash_exp, how='inner')
        print(avg_res_dash_exp)
        print_seperator()

        # Avg result per power bi experience
        print("Average correctness per Power BI experience (subset of dashboard experience)")
        print()
        df_agg = question_df.groupby(['power_bi_experience', 'user_id', 'answer_correct'])['answer_correct'].count().reset_index(name='count')
        avg_res_pbi_exp = df_agg.groupby(['power_bi_experience', 'answer_correct'])['count'].mean().reset_index(name='avg')
        avg_res_pbi_exp = avg_res_pbi_exp[avg_res_pbi_exp['answer_correct']].reset_index(drop=True)
        user_count_pbi_exp = question_df.groupby(['power_bi_experience'])['user_id'].nunique().reset_index(name='user_count')
        avg_res_pbi_exp = pd.merge(avg_res_pbi_exp, user_count_pbi_exp, how='inner')
        avg_res_pbi_exp = avg_res_pbi_exp[avg_res_pbi_exp['power_bi_experience'] > -1].reset_index(drop=True)
        print(avg_res_pbi_exp)
        print_seperator()

        # Avg result per confidence
        print("Average correctness per dashboard confidence [1-6 (6 - very high)] (subset of dashboard experience)")
        print()
        df_agg = question_df.groupby(['confidence', 'user_id', 'answer_correct'])['answer_correct'].count().reset_index(name='count')
        avg_res_confi = df_agg.groupby(['confidence', 'answer_correct'])['count'].mean().reset_index(name='avg')
        avg_res_confi = avg_res_confi[avg_res_confi['answer_correct']].reset_index(drop=True)
        user_count_pbi_exp = question_df.groupby(['confidence'])['user_id'].nunique().reset_index(name='user_count')
        avg_res_confi = pd.merge(avg_res_confi, user_count_pbi_exp, how='inner')
        avg_res_confi = avg_res_confi[avg_res_confi['confidence'] > -1].reset_index(drop=True)
        print(avg_res_confi)
        print_seperator()

        # gender
        print("Gender")
        print()
        gender_dist = user_df['gender'].value_counts()
        print(gender_dist)
        print_seperator()

        # age
        print("Age")
        print()
        print_list_item_overview([user.age for user in users])
        print_seperator()

        # satisfaction
        print("Satisfaction [1-6 (6 = very high)]:")
        print
        print_list_item_overview([user.satisfaction for user in users])

        ###########################
        ##  Question Evaluation  ##
        ###########################

        print_seperator()
        print("QUESTION EVALUATION")
        print_seperator()

        # Correct answers per question
        print("General evaluation of questions:")
        print()
        quest_eval = question_df.groupby(['question_id', 'answer_correct'])['question_id'].count().reset_index(name='corr_count')
        quest_eval = quest_eval[quest_eval['answer_correct']].reset_index(drop=True)
        quest_eval['corr_dist'] = quest_eval['corr_count']/total_no_of_users
        quest_eval = quest_eval.drop(columns='answer_correct')
        quest_eval = pd.merge(quest_eval, question_df.groupby(['question_id'])['mental_effort'].mean().reset_index(name='avg_effort'), how='inner')
        quest_eval = pd.merge(quest_eval, question_df.groupby(['question_id'])['no_of_steps'].mean().reset_index(name='avg_step_no'), how='inner')
        quest_eval = pd.merge(quest_eval, question_df.groupby(['question_id'])['no_of_steps'].min().reset_index(name='min_step_no'), how='inner')
        quest_eval = pd.merge(quest_eval, question_df.groupby(['question_id'])['no_of_steps'].max().reset_index(name='max_step_no'), how='inner')
        quest_eval = pd.merge(quest_eval, question_df.groupby(['question_id'])['running_time'].mean().reset_index(name='avg_time'), how='inner')
        quest_eval = pd.merge(quest_eval, question_df.groupby(['question_id'])['running_time'].min().reset_index(name='min_time'), how='inner')
        quest_eval = pd.merge(quest_eval, question_df.groupby(['question_id'])['running_time'].max().reset_index(name='max_time'), how='inner')
        print(quest_eval)
        print_seperator()

        # Experience & Satisfaction
        print("Experience, Satisfaction & Effort")
        print_seperator()
        # dashboard experience
        avg_sati_dash_exp = user_df.groupby(['dashboard_experience'])['satisfaction'].mean().reset_index(name='avg_satisfaction')
        avg_effort_dash_exp = question_df.groupby(['dashboard_experience'])['mental_effort'].mean().reset_index(name='avg_eff_total')
        no_user_dash_exp = user_df.groupby(['dashboard_experience'])['user_id'].count().reset_index(name='user_count')
        sum_dash_exp = pd.merge(avg_sati_dash_exp, avg_effort_dash_exp, how='inner')
        sum_dash_exp = pd.merge(sum_dash_exp, no_user_dash_exp, how='inner')
        print("Overview dashboard experience")
        print()
        print(sum_dash_exp)
        print_seperator()

        # power bi experience
        avg_sati_pbi_exp = user_df.groupby(['power_bi_experience'])['satisfaction'].mean().reset_index(name='avg_satisfaction')
        avg_effort_pbi_exp = question_df.groupby(['power_bi_experience'])['mental_effort'].mean().reset_index(name='avg_eff_total')
        no_user_pbi_exp = user_df.groupby(['power_bi_experience'])['user_id'].count().reset_index(name='user_count')
        sum_pbi_exp = pd.merge(avg_sati_pbi_exp, avg_effort_pbi_exp, how='inner')
        sum_pbi_exp = pd.merge(sum_pbi_exp, no_user_pbi_exp, how='inner')
        print("Overview power bi experience")
        print()
        print(sum_pbi_exp)
        print_seperator()