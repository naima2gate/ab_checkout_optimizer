# src/bandits.py
import numpy as np

class ThompsonBernoulli:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)
    
    def select_arm(self):
        sample = np.random.beta(self.successes+1, self.failures+1)
        return np.argmax(sample)
    
    def update(self, arm, reward):
        if reward > 0:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1

class ThompsonGaussian:
    def __init__(self, n_arms, mu0=0.0, sigma0=1.0):
        self.n_arms = n_arms
        self.mu = np.ones(n_arms) * mu0
        self.tau = np.ones(n_arms) / sigma0**2
        self.n = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
    
    def select_arm(self):
        sample = np.random.normal(self.mu, 1/np.sqrt(self.tau))
        return np.argmax(sample)
    
    def update(self, arm, reward):
        self.n[arm] += 1
        self.sums[arm] += reward
        self.tau[arm] = 1 + self.n[arm]
        self.mu[arm] = self.sums[arm] / self.tau[arm]

class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0
    
    def select_arm(self):
        self.total_counts += 1
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                return i
        ucb_values = self.values + np.sqrt(2*np.log(self.total_counts)/self.counts)
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n-1)*self.values[arm] + reward)/n

class EpsilonGreedy:
    def __init__(self, n_arms, epsilon=0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
    
    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        return np.argmax(self.values)
    
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] = ((n-1)*self.values[arm] + reward)/n
