from provectories.provectories import Provectories


provectories = Provectories()

for i in [1,2,3,4,5,6,7,8]:
    provectories.create_csv_file(f'question_{i}', i, user_cols=['dashboard_experience'], quest_cols=['answer_correct', 'no_of_steps', 'running_time'])

provectories.evaluate_data_sets()
print("finito")