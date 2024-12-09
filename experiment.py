import gymnasium as gym
from controllers.pd import PDController
from controllers.pd import EnergySwingUpController
from controllers.sampling import SamplingController


def run_task_3():
    """Zadanie 3 - PD Controller"""
    print("\n🔹 Uruchamianie Zadania 3: PD Controller 🔹")
    n_steps = 200
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    controller = PDController(env, kp=66.0, kd=12.0)  # Dostosuj wartości Kp i Kd

    observation, info = env.reset(options={"state": [3.1, 0.0]})  # Ustawienie początkowego stanu
    episode_reward = 0.0
    for i in range(n_steps):
        action = controller.compute_control(observation)  # Sterowanie oparte na PD Controllerze
        next_observation, reward, terminated, truncated, info = env.step(action)
        observation = next_observation
        episode_reward += reward
        env.render()

    print(f"✅ Zakończono Zadanie 3 — Skumulowana nagroda: {episode_reward}")


def run_task_4():
    """Zadanie 4 - Energy Swing-Up Controller"""
    print("\n🔹 Uruchamianie Zadania 4: Energy Swing-Up Controller 🔹")
    n_steps = 500  # Wydłużony czas, aby dać szansę na "swing-up"
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    controller = EnergySwingUpController(env, kp=66.0, kd=12.0, energy_target=2.0)  # Ustaw poziom energii

    observation, info = env.reset(options={"state": [3.1, 0.0]})  # Start z pozycji bliskiej "w dół"
    episode_reward = 0.0
    for i in range(n_steps):
        action = controller.compute_control(observation)  # Sterowanie oparte na Energy Swing-Up Controllerze
        next_observation, reward, terminated, truncated, info = env.step(action)
        observation = next_observation
        episode_reward += reward
        env.render()

    print(f"✅ Zakończono Zadanie 4 — Skumulowana nagroda: {episode_reward}")


def run_task_5():
    """Zadanie 5 - Sampling Controller"""
    print("\n🔹 Uruchamianie Zadania 5: Sampling Controller 🔹")
    n_steps = 200
    env = gym.make("Pendulum-v1", render_mode="human", g=9.81)
    controller = SamplingController(env, num_rollouts=100, rollout_length=10)  # Parametry sampling controller

    observation, info = env.reset(options={"state": [3.1, 0.0]})  # Start z pozycji bliskiej "w dół"
    episode_reward = 0.0
    for i in range(n_steps):
        action = controller.compute_control(observation)  # Sterowanie oparte na Sampling Controllerze
        next_observation, reward, terminated, truncated, info = env.step(action)
        observation = next_observation
        episode_reward += reward
        env.render()

    print(f"✅ Zakończono Zadanie 5 — Skumulowana nagroda: {episode_reward}")


if __name__ == "__main__":
    print("\n📘 Wybierz zadanie do uruchomienia:")
    print("3 - Zadanie 3 (PD Controller)")
    print("4 - Zadanie 4 (Energy Swing-Up Controller)")
    print("5 - Zadanie 5 (Sampling Controller)")
    print("0 - Uruchom wszystkie zadania po kolei\n")

    choice = input("🔷 Wybór (0, 3, 4, 5): ")

    if choice == '3':
        run_task_3()
    elif choice == '4':
        run_task_4()
    elif choice == '5':
        run_task_5()
    elif choice == '0':
        run_task_3()
        run_task_4()
        run_task_5()
    else:
        print("❌ Nieprawidłowy wybór. Zakończenie programu.")
