from provectories.provectories import Provectories

provectories = Provectories()
for user in provectories.users:
    print(user)
    for question in user.questions:
        print(question)
        # for state in question.states:
        #     print(state)

# for i in [1, 2, 3, 4, 5, 6, 7, 8]:
#     provectories.create_csv_file(f"question_{i}", i, [
#         "answer_correct",
#         "running_time",
#         "triggeredAction",
#         "selectedValues",
#         "filteredValues"
#     ])