import gymnasium as gym
import torch
import torch.nn as nn
import pickle
from torch.distributions import Normal


# ==========================================
# 1. ВОССТАНАВЛИВАЕМ АРХИТЕКТУРУ СЕТИ
# ==========================================
# PyTorch сохраняет только веса, поэтому класс должен быть объявлен точно так же
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.actor_mean = nn.Linear(64, action_dim)
        self.actor_log_std = nn.Linear(64, action_dim)
        self.critic_value = nn.Linear(64, 1)

    def forward(self, state):
        x = self.shared(state)
        mean = self.actor_mean(x)
        log_std = self.actor_log_std(x)
        log_std_clamped = torch.clamp(log_std, min=-5, max=2)
        std = log_std_clamped.exp() + 1e-8
        dist = Normal(mean, std)
        value = self.critic_value(x)
        return dist, value


# ==========================================
# 2. ЗАГРУЗКА МОДЕЛИ И СРЕДЫ
# ==========================================
state_dim = 2
action_dim = 1

# Инициализируем модель и загружаем в нее сохраненные веса
model = ActorCritic(state_dim, action_dim)
model.load_state_dict(torch.load("ppo_mcc.pth"))
model.eval()  # Переводим в режим тестирования (отключаем обучение)

# Создаем среду с окном отрисовки
env = gym.make("MountainCarContinuous-v0", render_mode="human")

# Навешиваем обертку нормализации (как при обучении)
env = gym.wrappers.NormalizeObservation(env)

# КРИТИЧЕСКИ ВАЖНО: Подгружаем сохраненную статистику нормализации!
with open("ppo_obs_rms.pkl", "rb") as f:
    env.obs_rms = pickle.load(f)

# ==========================================
# 3. ВИЗУАЛИЗАЦИЯ
# ==========================================
print("Запуск визуализации...")
successes = 0

for episode in range(5):  # Посмотрим 5 эпизодов
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.FloatTensor(obs).unsqueeze(0)

        with torch.no_grad():
            dist, _ = model(state_tensor)
            # Берем среднее значение (детерминированно) и пропускаем через tanh
            action = torch.tanh(dist.mean).numpy()[0]

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        if terminated:
            successes += 1

    print(f"Эпизод {episode + 1} завершен. Награда: {total_reward:.2f}")

env.close()
print(f"\nУспешных эпизодов: {successes}/5")