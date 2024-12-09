import gymnasium as gym
from controllers.pd import EnergySwingUpController

n_steps = 500  # Wydłużmy czas, aby dać szansę na "swing-up"
env = gym.make("Pendulum-v1", render_mode="human", g=9.81)

# Używamy kontrolera Swing-up (energetycznego) zamiast prostego PD
controller = EnergySwingUpController(env, kp=66.0, kd=8.0, energy_target=1.2)  # Dostosuj poziom energii

# USTAWIENIE STANU POCZĄTKOWEGO
observation, info = env.reset(options={"state": [3.1, 0.0]})  # Start z bliskiej pozycji "w dół"

episode_reward = 0.0
for i in range(n_steps):
    action = controller.compute_control(observation)  # Użycie stanu jako argumentu
    next_observation, reward, terminated, truncated, info = env.step(action)
    observation = next_observation  # Aktualizacja stanu
    episode_reward += reward
    env.render()
print(f"Episode reward: {episode_reward}")
