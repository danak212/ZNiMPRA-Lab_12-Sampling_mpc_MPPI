import numpy as np
from controllers.interface import Controller


class SamplingController(Controller):
    def __init__(self, env, num_rollouts=100, rollout_length=10):
        super().__init__(env)
        self.num_rollouts = num_rollouts  # Liczba prób
        self.rollout_length = rollout_length  # Długość symulacji każdej trajektorii

    def compute_control(self, state):
        """
        Oblicza sterowanie w oparciu o aktualny stan.
        Metoda samplingowa testuje różne losowe ciągi akcji.
        """
        best_reward = -np.inf
        best_action = 0

        for i in range(self.num_rollouts):
            total_reward = 0
            observation = state.copy()
            action_sequence = np.random.uniform(-2, 2, size=(self.rollout_length, 1))

            for action in action_sequence:
                next_observation, reward, terminated, truncated, info = self.env.step([action])
                total_reward += reward
                observation = next_observation

                if terminated or truncated:
                    break

            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action_sequence[0]  # Wybierz pierwszą akcję z najlepszej trajektorii

        # Ograniczenie momentu do zakresu sterowania środowiska
        max_torque = self.env.action_space.high[0]  # Maksymalny moment
        best_action = np.clip(best_action, -max_torque, max_torque)
        return [best_action]  # Zwracamy jako lista (wymaga tego step())
