import gymnasium as gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F

# ==========================================
# 1. ГИПЕРПАРАМЕТРЫ
# ==========================================
# Скорость обучения
LEARNING_RATE = 1e-05
# Коэффициент дисконтирования Gamma - определяет, насколько агент ценит будущие награды
GAMMA = 0.9999
# Количество шагов на среду
N_STEPS = 250
# Общее количество циклов обучения
NUM_ITERATIONS = 800
# Энтропия. Заставляет агента исследовать пространство действий
ENTROPY_COEF = 0.05
# Вес ошибки Критика
VALUE_COEF = 0.5
# Параметр GAE (Generalized Advantage Estimation Lambda):
GAE_LAMBDA = 0.95
# Количество параллельных сред
NUM_ENVS = 4
# Количество эпох
N_EPOCHS = 10
# Количество примеров, на которых считается градиент за один шаг.
BATCH_SIZE = 128
# Главная фишка PPO. Запрещает политике меняться более чем на N% за один шаг.
CLIP_EPSILON = 0.2


# ==========================================
# 2. НЕЙРОСЕТЬ
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Используем Tanh для стабильности физических сред
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
# 3. СРЕДА
# ==========================================
def make_env():
    env = gym.make("MountainCarContinuous-v0")
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.NormalizeReward(env, gamma=GAMMA)
    return env


envs = gym.vector.SyncVectorEnv([make_env for _ in range(NUM_ENVS)])
state_dim = envs.single_observation_space.shape[0]
action_dim = envs.single_action_space.shape[0]

model = ActorCritic(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

episode_rewards = []
current_episode_rewards = np.zeros(NUM_ENVS)
obs, _ = envs.reset()

# ==========================================
# 4. ЦИКЛ ОБУЧЕНИЯ
# ==========================================
for iteration in range(NUM_ITERATIONS):
    states, actions, rewards, log_probs, values, dones = [], [], [], [], [], []

    for _ in range(N_STEPS):
        obs_tensor = torch.FloatTensor(obs)
        with torch.no_grad():
            dist, value = model(obs_tensor)
            x_t = dist.rsample()
            action = torch.tanh(x_t)

            log_prob_normal = dist.log_prob(x_t)
            log_prob_tanh = torch.log(1 - action.pow(2) + 1e-6)
            log_prob = (log_prob_normal - log_prob_tanh).sum(dim=-1, keepdim=True)

        next_obs, reward, terminated, truncated, _ = envs.step(action.numpy())
        done_step = np.logical_or(terminated, truncated)

        states.append(obs_tensor)
        actions.append(x_t)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1))
        log_probs.append(log_prob)
        values.append(value)
        dones.append(torch.FloatTensor(done_step).unsqueeze(1))

        obs = next_obs
        current_episode_rewards += reward

        for i in range(NUM_ENVS):
            if done_step[i]:
                episode_rewards.append(current_episode_rewards[i])
                current_episode_rewards[i] = 0

    states = torch.cat(states)
    actions = torch.cat(actions)
    rewards = torch.cat(rewards)
    old_log_probs = torch.cat(log_probs)
    values = torch.cat(values)
    dones = torch.cat(dones)

    with torch.no_grad():
        _, next_value = model(torch.FloatTensor(obs))

    rewards = rewards.view(N_STEPS, NUM_ENVS, 1)
    values_reshaped = values.view(N_STEPS, NUM_ENVS, 1)
    dones_reshaped = dones.view(N_STEPS, NUM_ENVS, 1)

    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(NUM_ENVS, 1)

    for t in reversed(range(N_STEPS)):
        if t == N_STEPS - 1:
            next_val = next_value
        else:
            next_val = values_reshaped[t + 1]

        mask = 1.0 - dones_reshaped[t]
        delta = rewards[t] + GAMMA * next_val * mask - values_reshaped[t]
        gae = delta + GAMMA * GAE_LAMBDA * mask * gae
        advantages[t] = gae

    advantages = advantages.view(N_STEPS * NUM_ENVS, 1)
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    dataset_size = N_STEPS * NUM_ENVS
    indices = np.arange(dataset_size)

    for epoch in range(N_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            idx = indices[start:end]

            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_advantages = advantages[idx]
            batch_returns = returns[idx]

            dist, new_values = model(batch_states)
            new_log_prob_normal = dist.log_prob(batch_actions)
            batch_actions_tanh = torch.tanh(batch_actions)
            new_log_prob_tanh = torch.log(1 - batch_actions_tanh.pow(2) + 1e-6)
            new_log_probs = (new_log_prob_normal - new_log_prob_tanh).sum(dim=-1, keepdim=True)

            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - CLIP_EPSILON, 1.0 + CLIP_EPSILON) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

            critic_loss = F.smooth_l1_loss(new_values, batch_returns)
            loss = actor_loss + VALUE_COEF * critic_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

    if (iteration + 1) % 10 == 0 and len(episode_rewards) > 0:
        avg_reward = np.mean(episode_rewards[-20:])
        print(f"Iter {iteration + 1}, Avg Reward (last 20 eps): {avg_reward:.2f} | Total eps: {len(episode_rewards)}")

# ==========================================
# 5. ТЕСТИРОВАНИЕ
# ==========================================
print("\n--- ТЕСТИРОВАНИЕ ---")
test_env = gym.make("MountainCarContinuous-v0")
test_env = gym.wrappers.NormalizeObservation(test_env)

# ВОТ ОНА, МАГИЯ: Копируем среднее и дисперсию из тренировочной среды!
# envs.envs[0] - это первая среда из нашего SyncVectorEnv
test_env.obs_rms = envs.envs[0].env.obs_rms  # type: ignore

successes = 0
for _ in range(10):
    state, _ = test_env.reset()
    done = False
    total_reward = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, _ = model(state_tensor)
            action = torch.tanh(dist.mean)

        state, reward, terminated, truncated, _ = test_env.step(action.numpy()[0])
        done = terminated or truncated
        total_reward += reward
        if terminated:
            successes += 1
    print(f"Test Episode Reward: {total_reward:.2f}")

print(f"\nУспешных эпизодов: {successes}/10")

print("\n--- СОХРАНЕНИЕ ---")
# 1. Сохраняем веса нейросети
torch.save(model.state_dict(), "ppo_mcc.pth")

# 2. Сохраняем статистику нормализации (obs_rms)
# envs.envs[0] - это NormalizeReward, .env - это NormalizeObservation
obs_rms_to_save = envs.envs[0].env.obs_rms  # type: ignore
with open("ppo_obs_rms.pkl", "wb") as f:
    pickle.dump(obs_rms_to_save, f)

print("Модель (ppo_mcc.pth) и статистика (ppo_obs_rms.pkl) успешно сохранены!")
