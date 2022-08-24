from provectories.provectories import Provectories


provectories = Provectories()

for i in [1,2,3,4,5,6,7,8]:
    provectories.create_csv_file(f'question_{i}_year_weight_35', i, user_cols=['dashboard_experience', 'power_bi_experience', 'confidence', 'sample'], quest_cols=['answer_correct', 'no_of_steps'])

provectories.evaluate_data_sets()
print("finito")