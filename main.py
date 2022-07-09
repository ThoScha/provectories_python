from provectories.stories import Stories

for i in [1, 2, 3, 4, 5, 6, 7, 8]:
    s = Stories(i)
    s.calculateDistance()
    s.writeCSV(f"question_{i}")
