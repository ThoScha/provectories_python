class State:
    def __init__(self, timestamp, triggered_action, selected_values, filtered_values, feature_vector):
        self.timestamp = timestamp
        self.triggered_action = triggered_action
        self.selected_values = selected_values
        self.filtered_values = filtered_values
        self.feature_vector = feature_vector

    def __str__(self):
        return self.timestamp + " " + self.feature_vector

    def __getitem__(self, key):
        return getattr(self, key)
        