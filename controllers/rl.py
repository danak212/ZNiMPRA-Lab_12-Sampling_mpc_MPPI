import numpy as np
from controllers.interface import Controller


class RLController(Controller):
    def __init__(self, env):
        super().__init__(env)
        # Parametry RL (to jest przykładowy algorytm Q-Learning)
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1  # Eksploracja
        self.q_table = {}  # Prosta Q-tablica do przechowywania wartości akcji

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = max([self.get_q_value(next_state, a) for a in [-2, 2]])
        current_q = self.get_q_value(state, action)
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * best_next_action)
        self.q_table[(tuple(state), action)] = new_q

    def compute_control(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.uniform(-2, 2)  # Eksploracja
        else:
            q_values = {action: self.get_q_value(state, action) for action in [-2, 2]}
            action = max(q_values, key=q_values.get)  # Wybierz najlepszą akcję
        return [action]  # Wymaga listy w step()
