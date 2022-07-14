from provectories.provectories import Provectories

provectories = Provectories()
provectories.create_csv_file('test_file_1', 3, user_cols=['dashboard_experience'], quest_cols=['answer_correct', 'no_of_steps', 'running_time'])

print("finito")