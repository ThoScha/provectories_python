class State:
    def __init__(self, timestamp, triggered_action, selected_values, filtered_values, feature_vector):
        self.timestamp = timestamp
        self.triggered_action = triggered_action
        self.selected_values = selected_values
        self.filtereValues = filtered_values
        self.feature_vector = feature_vector

    def __str__(self):
        return self.timestamp + " " + self.feature_vector
        