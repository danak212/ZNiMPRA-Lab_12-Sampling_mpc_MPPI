import numpy as np
from controllers.interface import Controller


class PDController(Controller):
    def __init__(self, env, kp=10.0, kd=1.0):
        super().__init__(env)
        self.kp = kp  # Wzmocnienie proporcjonalne
        self.kd = kd  # Wzmocnienie różniczkujące

    def compute_control(self, state):
        """
        Oblicza sterowanie w oparciu o aktualny stan.
        Stan jest listą [cos(theta), sin(theta), theta_dot], gdzie:
        - cos(theta), sin(theta) to kosinus i sinus kąta
        - theta_dot to prędkość kątowa
        """
        cos_theta, sin_theta, theta_dot = state  # Rozpakowanie stanu
        theta = np.arctan2(sin_theta, cos_theta)  # Odzyskujemy kąt theta
        # Sterowanie oparte na równaniu PD
        control = -self.kp * theta - self.kd * theta_dot
        # Ograniczenie momentu do zakresu sterowania środowiska
        max_torque = self.env.action_space.high[0]  # Maksymalny moment
        control = np.clip(control, -max_torque, max_torque)
        return [control]  # Zwracamy jako lista (wymaga tego step())
