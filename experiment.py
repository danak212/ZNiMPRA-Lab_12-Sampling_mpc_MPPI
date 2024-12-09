import gymnasium as gym
from controllers.pd import PDController

n_steps = 200
env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
controller = PDController(env, kp=66.0, kd=12.0)  # Dostosuj wartości Kp i Kd

# USTAWIENIE STANU POCZĄTKOWEGO
observation, info = env.reset(options={"state": [3.1, 0.0]})  # Losowy start

episode_reward = 0.0
for i in range(n_steps):
    action = controller.compute_control(observation)  # Użycie stanu jako argumentu
    next_observation, reward, terminated, truncated, info = env.step(action)
    observation = next_observation  # Aktualizacja stanu
    episode_reward += reward
    env.render()
print(f"Episode reward: {episode_reward}")
