import numpy as np


class QlearningAgent:
    def __init__(self, alpha=0.5, epsilon=0.1, discount_factor=0.99, num_states=100, num_actions=4):
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # exploration rate
        self.discount_factor = discount_factor  # discount factor
        self.q_table = np.zeros((num_states, num_actions))  # Q-table initialization

    def update_q_table(self, state, action, reward, next_state):
        # Calculate the maximum Q-value for the next state
        max_q_next_state = np.max(self.q_table[next_state])

        # Update the Q-value for the current state and action
        self.q_table[state][action] += self.alpha * (
                    reward + self.discount_factor * max_q_next_state - self.q_table[state][action])

    def choose_action(self, state):
        # Choose a random action with probability epsilon
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(self.q_table.shape[1])
        # Choose the action with the highest Q-value for the current state
        else:
            action = np.argmax(self.q_table[state])
        return action
