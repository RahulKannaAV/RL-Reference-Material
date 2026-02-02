import numpy as np
import gym
from gym import Env, spaces

class CustomFrozenLakeEnv(Env):
    def __init__(self, grid_size=4):
        super(CustomFrozenLakeEnv, self).__init__()

        #define grid size
        self.grid_size = grid_size

        # action spaces : 0 - left ; 1 - down ; 2 - right ; 3 - up
        self.action_space = spaces.Discrete(4)

        # observation space where each state corresponds to a grid cell
        self.observation_space = spaces.Discrete(grid_size * grid_size)

        # define lake layout : 0 = frozen, 1 = hole, 2 = goal
        self.lake = np.zeros((grid_size, grid_size), dtype=int)
        self.lake[-1, -1] = 2  # Goal
        self.lake[1, 1] = 1  # Hole
        self.lake[2, 2] = 1  # Hole

        #rewards
        self.rewards = np.zeros_like(self.lake, dtype=float)
        self.rewards[-1, -1] = 1.0  # reward
        self.rewards[self.lake == 1] = -1.0  # penalty

        
        #starting state
        self.state = (0, 0)


    def step(self, action):
        x, y = self.state #read current state

        # move agent
        if action == 0:  # Left
            y = max(0, y - 1)
        elif action == 1:  # Down
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:  # Right
            y = min(self.grid_size - 1, y + 1)
        elif action == 3:  # Up
            x = max(0, x - 1)

        # state updation
        self.state = (x, y)

        reward = self.rewards[x, y]

        # termination check
        done = self.lake[x, y] == 2 or self.lake[x, y] == 1

        # return all info
        return self._get_state_index(), reward, done, {}

    def reset(self):
        self.state = (0, 0)
        return self._get_state_index()

    def render(self):
        grid = np.array(self.lake, dtype=str)
        grid[self.lake == 0] = "."
        grid[self.lake == 1] = "H"
        grid[self.lake == 2] = "G"
        x, y = self.state
        grid[x, y] = "A"  # agent
        print("\n".join(" ".join(row) for row in grid))

    def _get_state_index(self):
        return self.state[0] * self.grid_size + self.state[1]



env = CustomFrozenLakeEnv(grid_size=4)
state = env.reset()
done = False

print("Initial Environment:")
env.render()

while not done:
    action = env.action_space.sample()  # random policy
    next_state, reward, done, info = env.step(action)
    print(f"\nAction: {action}")
    env.render()
    print(f"Reward: {reward}")