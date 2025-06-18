
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display, clear_output
from sklearn.preprocessing import KBinsDiscretizer
from collections import defaultdict
import random
import numpy as np
import imageio
import cv2
import os
import time
import json
import csv
from math import floor, log10, ceil
from pandas import Series
from IPython.display import Video, display

# %%
import gymnasium as gym
env = gym.make("LunarLander-v3")
print(env.unwrapped.__class__.__module__)



class QLearningAgent:
    def __init__(self, n_bins, lower_bounds, upper_bounds, bin_strategies, n_actions,
                 learning_coeff=1.0, learning_decay=0.00016,
                 exploration_coeff=0.005, exploration_decay=1e-6,
                 min_learn=0, min_explore=0, discount=1):
        # Khởi tạo các tham số chính
        self.n_bins = n_bins  # số lượng bin (khoảng rời rạc) cho mỗi chiều trạng thái
        self.lower_bounds = lower_bounds  # giá trị nhỏ nhất cho mỗi chiều trạng thái
        self.upper_bounds = upper_bounds  # giá trị lớn nhất cho mỗi chiều trạng thái
        self.bin_strategies = bin_strategies  # chiến lược rời rạc hóa ('U' hoặc 'u')
        self.n_actions = n_actions  # số lượng hành động trong môi trường

        # Các siêu tham số của thuật toán
        self.learning_coeff = learning_coeff  # tốc độ học ban đầu (learning rate)
        self.learning_decay = learning_decay  # tốc độ giảm learning rate theo episode
        self.exploration_coeff = exploration_coeff  # tỉ lệ khám phá ban đầu (epsilon)
        self.exploration_decay = exploration_decay  # tốc độ giảm tỉ lệ khám phá theo episode
        self.min_learn = min_learn  # learning rate tối thiểu
        self.min_explore = min_explore  # tỉ lệ khám phá tối thiểu
        self.discount = discount  # hệ số giảm giá (discount factor)

        # Khởi tạo bảng Q với kích thước dựa trên số bin và số hành động
        self.Q_table = np.zeros(self.n_bins + (self.n_actions,))
        # Từ điển map chiến lược rời rạc đến hàm tương ứng
        self.bin_funcs = {'U': self.get_bin_U, 'u': self.get_bin_u}

    def get_bin_U(self, value, l, u, bins):
        # Giới hạn value trong khoảng [l, u]
        value = min(u, max(l, value))
        # Tính chỉ số bin rời rạc hóa với chiến lược 'U' (giới hạn giá trị)
        return int(min(np.floor((value - l) / (u - l) * bins), bins - 1))

    def get_bin_u(self, value, l, u, bins):
        # Giới hạn value trong khoảng [l, u]
        value = min(u, max(l, value))
        # Tính chỉ số bin rời rạc hóa với chiến lược 'u' (không giới hạn giá trị đầu vào trước)
        return int(min(np.floor((value - l) / (u - l) * bins), bins - 1))

    def discretize(self, obs):
        # Rời rạc hóa vector trạng thái quan sát (obs) thành tuple các chỉ số bin
        return tuple(
            self.bin_funcs[strat](val, low, high, bins)
            for val, strat, bins, low, high
            in zip(obs, self.bin_strategies, self.n_bins, self.lower_bounds, self.upper_bounds)
        )

    def policy(self, state):
        # Chọn hành động tốt nhất theo Q_table tại trạng thái đã rời rạc hóa
        return np.argmax(self.Q_table[state])

    def exploration_rate(self, episode):
        # Tính tỉ lệ khám phá (epsilon) giảm dần theo số episode
        return max(self.min_explore,
                   self.exploration_coeff * pow(1 - self.exploration_decay, episode))

    def learning_rate(self, episode):
        # Tính tốc độ học (learning rate) giảm dần theo số episode
        return max(self.min_learn,
                   self.learning_coeff * pow(1 - self.learning_decay, episode))

    def update(self, state, action, reward, next_state, episode):
        # Cập nhật bảng Q dựa trên công thức Q-learning
        lr = self.learning_rate(episode)  # learning rate tại episode hiện tại
        future_optimal_value = np.max(self.Q_table[next_state])  # giá trị Q tốt nhất ở trạng thái kế tiếp
        learned_value = reward + self.discount * future_optimal_value  # giá trị Q được cập nhật
        old_value = self.Q_table[state][action]  # giá trị Q cũ
        # Cập nhật giá trị Q mới theo công thức
        self.Q_table[state][action] = (1 - lr) * old_value + lr * learned_value
        


def adjust_reward(reward, state, done):
    if done and reward >= 100:
        reward += 50
    elif done and reward <= -100:
        reward -= 50
    if done and (state[6] == 0 or state[7] == 0):
        reward -= 75
    return reward



# %%
def create_episode_video_with_reward_plot(env, agent, episode_num, max_steps=1000, save_dir=None, fps=30):
    # Khởi tạo môi trường, lấy trạng thái đầu tiên và rời rạc hóa trạng thái
    obs, info = env.reset()
    state = agent.discretize(obs)
    terminated, truncated = False, False

    total_reward = 0  # Tổng phần thưởng tích lũy của episode
    reward_history = []  # Lịch sử tổng phần thưởng qua các bước

    # Lấy khung hình đầu tiên của môi trường để xác định kích thước video
    frame_env = env.render()
    height_env, width_env, _ = frame_env.shape

    # Chiều cao và chiều rộng dùng cho biểu đồ reward, lấy 1/3 chiều cao của môi trường
    plot_height = height_env // 3
    plot_width = width_env

    # Định dạng video (codec mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Nếu không có thư mục lưu video, dùng thư mục hiện tại
    if save_dir is None:
        save_dir = os.getcwd()
    os.makedirs(save_dir, exist_ok=True)

    # Đường dẫn lưu file video theo episode, định dạng mp4
    video_path = os.path.join(save_dir, f"episode_{str(episode_num).zfill(4)}.mp4")

    # Khởi tạo đối tượng ghi video, kích thước bao gồm khung môi trường và biểu đồ reward
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width_env, height_env + plot_height))

    # Vòng lặp chạy từng bước trong episode
    for step in range(max_steps):
        # Lấy hành động theo chính sách hiện tại của agent
        action = agent.policy(state)
        # Thực hiện hành động trong môi trường, lấy kết quả quan sát, reward và trạng thái kết thúc
        obs, reward, terminated, truncated, info = env.step(action)
        # Rời rạc hóa trạng thái quan sát mới
        next_state = agent.discretize(obs)
        state = next_state

        # Cộng dồn phần thưởng và lưu lịch sử
        total_reward += reward
        reward_history.append(total_reward)

        # Lấy khung hình môi trường hiện tại
        frame_env = env.render()

        # Vẽ biểu đồ tổng phần thưởng tích lũy
        fig, ax = plt.subplots(figsize=(plot_width / 100, plot_height / 100), dpi=100)
        ax.plot(reward_history, color='blue')  # đường cong reward
        ax.set_xlim(0, max_steps)
        ymin = min(0, min(reward_history)) - 1
        ymax = max(reward_history) + 1
        ax.set_ylim(ymin, ymax)
        ax.set_title('Total Reward Over Steps')
        ax.set_xlabel('Step')
        ax.set_ylabel('Total Reward')
        ax.grid(True)
        fig.tight_layout()

        # Chuyển biểu đồ matplotlib thành mảng ảnh
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf, size = fig.canvas.print_to_buffer()
        plot_img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
        plt.close(fig)  # đóng figure để tránh tràn bộ nhớ

        # Chuyển ảnh biểu đồ từ RGBA sang BGR (định dạng OpenCV)
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        # Resize ảnh biểu đồ cho vừa kích thước đã định
        plot_img = cv2.resize(plot_img, (plot_width, plot_height))

        # Nối dọc khung hình môi trường và biểu đồ reward thành một frame duy nhất
        combined_frame = np.vstack((frame_env, plot_img))
        # Ghi frame vào file video
        video_writer.write(combined_frame)

        # Nếu kết thúc episode (thất bại hoặc thành công), thoát vòng lặp
        if terminated or truncated:
            break

    # Giải phóng tài nguyên ghi video
    video_writer.release()
    print(f"Saved episode video: {video_path}")


# %%
def train_agent(env, agent, n_episodes, capture_eps, videosPath, export_fps=30, reward_modifier=None , record_video=False):
    rewards_all = []  # Danh sách lưu tổng phần thưởng mỗi episode

    # Vòng lặp chạy qua tất cả các episode
    for i in range(n_episodes):
        # Nếu episode hiện tại nằm trong danh sách cần lưu video
        if record_video and (i in capture_eps):
            # Tạo video ghi lại quá trình chạy của agent trong episode này
            create_episode_video_with_reward_plot(env, agent, i, 1000, videosPath, export_fps)
            print(f".", end="")  # In dấu chấm thể hiện tiến trình

        # Khởi tạo lại môi trường, nhận trạng thái ban đầu và rời rạc hóa
        obs, info = env.reset()
        state = agent.discretize(obs)
        terminated, truncated = False, False  # Cờ kết thúc episode
        total_reward = 0  # Tổng phần thưởng của episode hiện tại

        # Vòng lặp chạy từng bước trong episode cho tới khi kết thúc
        while not (terminated or truncated):
            # Chọn hành động: theo epsilon-greedy (thăm dò hoặc theo chính sách)
            if np.random.random() < agent.exploration_rate(i):
                action = env.action_space.sample()  # Chọn ngẫu nhiên (khám phá)
            else:
                action = agent.policy(state)  # Chọn hành động tốt nhất theo Q-table

            # Thực hiện hành động, nhận quan sát mới, phần thưởng và trạng thái kết thúc
            obs, reward, terminated, truncated, info = env.step(action)
            # Điều chỉnh phần thưởng nếu có hàm reward_modifier truyền vào
            reward_mod = reward_modifier(reward, obs, terminated or truncated) if reward_modifier else reward

            # Rời rạc hóa trạng thái mới
            next_state = agent.discretize(obs)
            # Cập nhật bảng Q dựa trên quan sát mới và phần thưởng
            agent.update(state, action, reward_mod, next_state, i)
            state = next_state  # Cập nhật trạng thái hiện tại

            total_reward += reward_mod  # Cộng dồn phần thưởng đã điều chỉnh

        # Lưu tổng phần thưởng của episode vào danh sách
        rewards_all.append(total_reward)

    # Trả về danh sách tổng phần thưởng qua tất cả episode
    return rewards_all



# ----- CSV Saver -----
def save_rewards_csv(rewards, name_tag, save_dir):
    csv_path = os.path.join(save_dir, f"rewards_{name_tag}.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward"])
        for i, reward in enumerate(rewards):
            writer.writerow([i + 1, reward])




# %%
def plot_reward_single(rewards, imagesPath, seed, title, filename, window_size=300):
    x = np.arange(len(rewards))
    y = np.array(rewards)
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.plot(x, y, color='blue', label="Tổng Reward mỗi Episode") # Vẽ đồ thị reward từng episode
     
     # Tính trung bình trượt (rolling average) trên window_size episode
    rolling_avg = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    plt.plot(x[window_size-1:], rolling_avg, color='red', label=f"Trung bình trượt {window_size} Episode") # Vẽ đồ thị trung bình trượt
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(imagesPath, filename))
    plt.close()
    print(f"✅ Đã lưu biểu đồ: {filename}")




# %%
def plot_reward_per_hyperparam(rewards, imagesPath, key, window_size=300):
    import matplotlib.pyplot as plt
    import numpy as np
    x = np.arange(len(rewards))
    y = np.array(rewards)
    plt.figure(figsize=(10, 6))
    plt.title(f"Reward cho Q-Learning - {key}")
    plt.plot(x, y, color='blue', label="Tổng Reward mỗi Episode")
    # Tính trung bình trượt (rolling average) với kích thước window_size
    rolling_avg = np.convolve(y, np.ones(window_size) / window_size, mode='valid')
    plt.plot(x[window_size - 1:], rolling_avg, color='red', label=f"Trung bình trượt {window_size} Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"reward_{key}.png"
    plt.savefig(os.path.join(imagesPath, filename))
    plt.close()
    print(f"✅ Đã lưu biểu đồ: {filename}")




# %%
def create_final_demo_video(env, agent, save_dir, filename="demo_final_episode.mp4", export_fps=30, seed=123):
    os.makedirs(save_dir, exist_ok=True)
    demo_video_path = os.path.join(save_dir, filename)

    obs, info = env.reset(seed=seed)
    state = agent.discretize(obs)
    terminated = False
    truncated = False

    frame = env.render()
    height, width, _ = frame.shape

    out_demo = cv2.VideoWriter(demo_video_path, cv2.VideoWriter_fourcc(*'mp4v'), export_fps, (width, height))

    while not (terminated or truncated):
        action = agent.policy(state)
        obs, reward, terminated, truncated, info = env.step(action)
        state = agent.discretize(obs)
        frame = env.render()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Chuyển màu trước khi ghi
            out_demo.write(frame_bgr)

    out_demo.release()
    print(f"✅ Video demo cuối cùng đã được lưu tại: {demo_video_path}")
    return demo_video_path


# %%
def display_video_html(path, width=600, height=400):
    video_tag = f"""
    <video width="{width}" height="{height}" controls>
      <source src="{path}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """
    display(HTML(video_tag))




# %%
# Config riêng cho version 1
SEEDS = [42069,69420]
n_episodes_v1 = 1000
n_bins_v1 = (6, 6, 6, 6, 6, 6, 2, 2)
lower_bounds_v1 = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0, 0.0]
upper_bounds_v1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0]
bin_strategies_v1 = "UUUUUUuu"


# %%

cwd = os.getcwd()
experimentPath = os.path.join(cwd, 'Experiments', 'Lunar Lander Compare', time.strftime("%Y-%m-%d_%H-%M-%S"))
os.makedirs(experimentPath, exist_ok=True)
capture_eps_v1 = list(range(0, n_episodes_v1, 100))

print(f"Folder lưu thí nghiệm: {experimentPath}")


# %%
# Config riêng cho version 1
SEEDS = [42069,69420]
n_episodes_v1 = 1000
n_bins_v1 = (6, 6, 6, 6, 6, 6, 2, 2)
lower_bounds_v1 = [-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0, 0.0]
upper_bounds_v1 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0]
bin_strategies_v1 = "UUUUUUuu"


SEED = SEEDS[0]

env = gym.make("LunarLander-v3", render_mode='rgb_array')
env.action_space.seed(SEED)
np.random.seed(SEED)

version_tag = "v1"

videosPath_v1 = os.path.join(experimentPath, version_tag, "video", f"seed_{SEED}")
imagesPath_v1 = os.path.join(experimentPath, version_tag, "img", f"seed_{SEED}")
os.makedirs(videosPath_v1, exist_ok=True)
os.makedirs(imagesPath_v1, exist_ok=True)

learning_coeff_v1 = 0.1
learning_decay_v1 = 1e-8
exploration_coeff_v1 = 0.05
exploration_decay_v1 = 1e-6
min_learn_v1 = 0
min_explore_v1 = 0
discount_v1 = 1

key_v1 = f"{version_tag}_lr_{learning_coeff_v1}_discount_{discount_v1}_seed_{SEED}"
print(f"Training {key_v1}")

agent_v1 = QLearningAgent(
    n_bins=n_bins_v1,
    lower_bounds=lower_bounds_v1,
    upper_bounds=upper_bounds_v1,
    bin_strategies=bin_strategies_v1,
    n_actions=env.action_space.n,
    learning_coeff=learning_coeff_v1,
    learning_decay=learning_decay_v1,
    exploration_coeff=exploration_coeff_v1,
    exploration_decay=exploration_decay_v1,
    min_learn=min_learn_v1,
    min_explore=min_explore_v1,
    discount=discount_v1,
)


rewards_v1 = train_agent(env, agent_v1, n_episodes_v1, capture_eps_v1, videosPath_v1, record_video=True)
save_rewards_csv(rewards_v1, key_v1, experimentPath)
plot_reward_single(rewards_v1, imagesPath_v1, key_v1, f"Reward version 1 - {key_v1}", f"reward_{key_v1}.png", window_size=100)


video_path = create_final_demo_video(env, agent_v1, videosPath_v1, filename=f"demo_final_v1_seed_{SEED}.mp4")

env.close()

display_video_html(video_path, width=600, height=400)
print(f"--- Done version 1 SEED {SEED} ---")


# %%
frame = env.render()
print(type(frame), frame.shape if frame is not None else None)


# %%
SEEDS = [42069, 69420]
n_episodes = 18000
n_bins = (4, 4, 4, 4, 4, 4, 2, 2)
lower_bounds = [-0.1, -0.2, -0.2, -0.2, -0.2, -0.2, 0., 0.]
upper_bounds = [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 1., 1.]

# %%
def train_version_final_reward_goc(SEED):
    env = gym.make("LunarLander-v3", render_mode='rgb_array')
    env.action_space.seed(SEED)
    np.random.seed(SEED)

    version_tag = "final"
    bin_strategies_final = "UUUUUUuu"
    videosPath_final = os.path.join(experimentPath, version_tag, "video", f"seed_{SEED}")
    imagesPath_final = os.path.join(experimentPath, version_tag, "img", f"seed_{SEED}")
    os.makedirs(videosPath_final, exist_ok=True)
    os.makedirs(imagesPath_final, exist_ok=True)

    learning_discount_pairs = [(1.0, 1.0), (0.5, 0.9)]

    for lr, discount in learning_discount_pairs:
        key_goc = f"{version_tag}_lr_{lr}_discount_{discount}_goc_seed_{SEED}"
        print(f"Training {key_goc}")

        agent_goc = QLearningAgent(n_bins, lower_bounds, upper_bounds, bin_strategies_final,
                                   env.action_space.n,
                                   learning_coeff=lr,
                                   discount=discount)
        rewards_goc = train_agent(env, agent_goc, n_episodes, capture_eps, videosPath_final, record_video=True)
        save_rewards_csv(rewards_goc, key_goc, experimentPath)
        plot_reward_single(rewards_goc, imagesPath_final, key_goc, f"Reward gốc - {key_goc}", f"reward_{key_goc}.png")
        video_path = create_final_demo_video(env, agent_goc, videosPath_final, filename=f"demo_final_{lr}_{discount}_goc_seed_{SEED}.mp4")
        display(Video(video_path, embed=True, width=600, height=400))

    env.close()
    print(f"--- Done final reward gốc SEED {SEED} ---")


# %%
train_version_final_reward_goc(SEEDS[0])

# %%
train_version_final_reward_goc(SEEDS[1])

# %%
def train_version_final_reward_moi(SEED):
    env = gym.make("LunarLander-v3", render_mode='rgb_array')
    env.action_space.seed(SEED)
    np.random.seed(SEED)

    version_tag = "final"
    bin_strategies_final = "UUUUUUuu"
    videosPath_final = os.path.join(experimentPath, version_tag, "video", f"seed_{SEED}")
    imagesPath_final = os.path.join(experimentPath, version_tag, "img", f"seed_{SEED}")
    os.makedirs(videosPath_final, exist_ok=True)
    os.makedirs(imagesPath_final, exist_ok=True)

    learning_discount_pairs = [(1.0, 1.0), (0.5, 0.9)]

    for lr, discount in learning_discount_pairs:
        key_moi = f"{version_tag}_lr_{lr}_discount_{discount}_moi_seed_{SEED}"
        print(f"Training {key_moi}")

        agent_moi = QLearningAgent(n_bins, lower_bounds, upper_bounds, bin_strategies_final,
                                  env.action_space.n,
                                  learning_coeff=lr,
                                  discount=discount)
        rewards_moi = train_agent(env, agent_moi, n_episodes, capture_eps, videosPath_final, reward_modifier=adjust_reward, record_video=False)
        save_rewards_csv(rewards_moi, key_moi, experimentPath)
        plot_reward_single(rewards_moi, imagesPath_final, key_moi, f"Reward điều chỉnh - {key_moi}", f"reward_{key_moi}.png")

    env.close()
    print(f"--- Done final reward điều chỉnh SEED {SEED} ---")




# %%
train_version_final_reward_moi(SEEDS[0])


# %%
train_version_final_reward_moi(SEEDS[1])


