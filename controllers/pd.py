from controllers.interface import Controller


class PDController(Controller):
    def __init__(self, env, kp=1.0, kd=0.1):
        super().__init__(env)
        self.kp = kp  # Wzmocnienie proporcjonalne
        self.kd = kd  # Wzmocnienie różniczkowe

    def compute_control(self, state):
        theta, theta_dot = state  # Odczytujemy kąt i prędkość kątową
        desired_theta = 0  # Chcemy, żeby wahadło było w pionie (0 radianów)
        error = desired_theta - theta
        control = self.kp * error - self.kd * theta_dot  # Prawo PD
        return [control]  # Zwracamy sterowanie w formie listy
