# %%
!apt-get install -y swig
!pip install box2d-py gym==0.26.2 moviepy --quiet

# %%
import gym
import random
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
from moviepy.editor import ImageSequenceClip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %%
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*samples))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)


# %%
def select_action(model, state, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(state).argmax().item()


# %%
def train():
    env = gym.make("LunarLander-v2")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(obs_dim, action_dim).to(device)
    target_net = DQN(obs_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    buffer = ReplayBuffer(100_000)

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.99
    batch_size = 128
    update_freq = 10
    max_episodes = 3000

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = select_action(policy_net, state, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(buffer) > batch_size:
                s, a, r, s2, d = buffer.sample(batch_size)
                s = torch.FloatTensor(s).to(device)
                a = torch.LongTensor(a).unsqueeze(1).to(device)
                r = torch.FloatTensor(r).unsqueeze(1).to(device)
                s2 = torch.FloatTensor(s2).to(device)
                d = torch.FloatTensor(d).unsqueeze(1).to(device)

                q_values = policy_net(s).gather(1, a)
                next_q_values = target_net(s2).max(1)[0].detach().unsqueeze(1)
                expected_q = r + (1 - d) * gamma * next_q_values

                loss = nn.MSELoss()(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if episode % update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode}: Total Reward = {total_reward:.2f}, Epsilon = {epsilon:.2f}")

    env.close()
    torch.save(policy_net.state_dict(), "/kaggle/working/dqn_lunarlander.pth")


# %%
def record_video(model_path="/kaggle/working/dqn_lunarlander.pth", output_file="/kaggle/working/lunarlander_dqn.mp4"):
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    model = DQN(env.observation_space.shape[0], env.action_space.n).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    frames = []
    state, _ = env.reset()
    done = False

    while not done:
        frame = env.render()
        frames.append(frame)
        action = select_action(model, state, epsilon=0.0, action_dim=env.action_space.n)
        next_state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state

    env.close()
    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(output_file, codec='libx264')


# %%
train()
record_video()


