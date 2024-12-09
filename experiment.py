import gymnasium as gym
from controllers.pd import PDController

n_steps = 200
env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
controller = PDController(env, kp=10.0, kd=1.0)  # Dostosuj wartości Kp i Kd

# USTAWIENIE STANU POCZĄTKOWEGO
# Ustawienie początkowego kąta na 180 stopni (pi radianów)
observation, info = env.reset(options={"state": [3.14, 0.0]})

episode_reward = 0.
for i in range(n_steps):
    action = controller.compute_control(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
print(f"Episode reward: {episode_reward}")
