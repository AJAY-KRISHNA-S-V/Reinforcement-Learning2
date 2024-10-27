import numpy as np
import pandas as pd
import gym
from gym import spaces
import matplotlib.pyplot as plt

# Define a custom trading environment
class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.n_steps = len(data)
        self.current_step = 0

        # Define action space (0: Hold, 1: Buy, 2: Sell)
        self.action_space = spaces.Discrete(3)

        # Define observation space (the stock price)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        self.balance = 1000  # Starting balance
        self.shares_held = 0
        self.total_profit = 0

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.shares_held = 0
        self.total_profit = 0
        return np.array([self.data[self.current_step]])

    def step(self, action):
        current_price = self.data[self.current_step]
        reward = 0

        if action == 1:  # Buy
            if self.balance >= current_price:
                self.shares_held += 1
                self.balance -= current_price
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price

        # Calculate total profit
        self.total_profit = self.balance + (self.shares_held * current_price) - 1000
        reward = self.total_profit  # Reward is the profit made

        self.current_step += 1
        done = self.current_step >= self.n_steps

        if done:
            return np.array([current_price]), reward, done, {}
        else:
            return np.array([self.data[self.current_step]]), reward, done, {}

# Load historical stock price data (example data)
data = pd.Series(np.random.normal(1, 0.02, 1000)).cumsum()  # Simulated stock prices

# Create the trading environment
env = TradingEnv(data.values)

# Parameters for Q-Learning
episodes = 1000
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Q-table initialization
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
q_table = np.zeros((1000, action_size))  # Limited state size for simplicity

# List to hold scores for plotting
scores = []

# Q-learning algorithm
for episode in range(episodes):
    state = env.reset()
    total_profit = 0
    done = False

    while not done:
        # Choose action (epsilon-greedy strategy)
        if np.random.rand() <= epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            state_index = min(int(state[0]), 999)  # Clip state index for simplicity
            action = np.argmax(q_table[state_index, :])  # Exploit learned values

        # Take action and observe result
        next_state, reward, done, _ = env.step(action)

        # Update Q-value
        state_index = min(int(state[0]), 999)  # Clip state index for simplicity
        next_state_index = min(int(next_state[0]), 999)  # Clip state index
        q_table[state_index, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state_index, :]) - q_table[state_index, action])

        total_profit += reward
        state = next_state

    scores.append(total_profit)  # Store the profit for this episode

    # Epsilon decay
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Epsilon: {epsilon:.2f}, Total Profit: {total_profit:.2f}")

# Plotting scores
plt.plot(scores)
plt.title('Total Profit Over Episodes (Trading Agent)')
plt.xlabel('Episodes')
plt.ylabel('Total Profit')
plt.grid()
plt.show()

env.close()
