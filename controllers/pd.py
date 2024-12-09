import numpy as np
from controllers.interface import Controller


class PDController(Controller):
    def __init__(self, env, kp, kd):
        super().__init__(env)
        self.kp = kp  # Wzmocnienie proporcjonalne
        self.kd = kd  # Wzmocnienie różniczkowe

    def compute_control(self, state):
        cos_theta, sin_theta, theta_dot = state  # Trzy elementy observation
        theta = np.arctan2(sin_theta, cos_theta)  # Przeliczamy kąt

        # Implementacja regulatora PD
        desired_theta = 0  # Chcemy, by wahadło stało pionowo
        error = desired_theta - theta  # Błąd pozycji
        d_error = -theta_dot  # Prędkość kątowa to "pochodna" kąta

        # Sterowanie zgodne z formułą PD
        action = self.kp * error + self.kd * d_error

        # Gym wymaga, by action było w formacie listy lub tablicy NumPy
        return [action]  # Zwracamy action jako listę
