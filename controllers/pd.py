import numpy as np
from controllers.interface import Controller


class EnergySwingUpController(Controller):
    def __init__(self, env, kp=66.0, kd=12.0, energy_target=10.0):
        super().__init__(env)
        self.kp = kp  # Wzmocnienie proporcjonalne (PD)
        self.kd = kd  # Wzmocnienie różniczkujące (PD)
        self.energy_target = energy_target  # Poziom energii potrzebnej do przełączenia na PD

    def compute_control(self, state):
        """
        Oblicza sterowanie w oparciu o aktualny stan.
        Stan jest listą [cos(theta), sin(theta), theta_dot], gdzie:
        - cos(theta), sin(theta) to kosinus i sinus kąta
        - theta_dot to prędkość kątowa
        """
        cos_theta, sin_theta, theta_dot = state  # Rozpakowanie stanu
        theta = np.arctan2(sin_theta, cos_theta)  # Odzyskujemy kąt theta

        # Obliczenie aktualnej energii
        kinetic_energy = 0.5 * (theta_dot ** 2)  # Energia kinetyczna (0.5 * m * v^2), masa = 1
        potential_energy = 1 - cos_theta  # Energia potencjalna (przesunięta do [0, 2] zamiast [-1, 1])
        total_energy = kinetic_energy + potential_energy

        # Sprawdzenie, czy energia jest wystarczająca, aby przełączyć się na tryb PD
        if total_energy >= self.energy_target and abs(theta) < np.deg2rad(30):
            # Gdy energia jest wystarczająca i wahadło jest blisko pionu -> przełącz na sterowanie PD
            control = -self.kp * theta - self.kd * theta_dot
        else:
            # Tryb "Swing-up" — kontrola energii
            desired_energy = self.energy_target
            energy_difference = desired_energy - total_energy
            control = 2 * energy_difference * np.sign(theta_dot * cos_theta)

        # Ograniczenie momentu do zakresu sterowania środowiska
        max_torque = self.env.action_space.high[0]  # Maksymalny moment
        control = np.clip(control, -max_torque, max_torque)
        return [control]  # Zwracamy jako lista (wymaga tego step())
