from provectories.provectories import Provectories

for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    s = Provectories(i)
    s.calculateDistance()
    s.writeCSV(f"question_{i}", [
        "answer_correct",
        "running_time",
        "triggeredAction",
        "selectedValues",
        "filteredValues"
    ])
