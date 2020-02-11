import numpy as np
import matplotlib.pyplot as plt

import gym


class CartPoleLinearPolicy: 

    def __init__(self, params):
        assert params.shape == (4,)
        self._params = params

    @property
    def params(self):
        return self._params.copy()

    def __call__(self, obs):
        '''
        TODO(Q1.1):

        Implement the binary linear policy.
        The policy should return 0 if the dot product of its parameters with the observation is negative, and 1 otherwise.
        '''
       # print (np.dot( obs.transpose(),  self.params))
        if    (  np.dot( self.params.transpose(),  obs) ) <0 :
            return 0
        else:
            return 1


def run_policy(policy, render=False):
    env = gym.make('CartPole-v1')
    obs = env.reset()

    rews = []
    while True:
        action = policy(obs)

        if render:
            env.render()

        obs, rew, done, _ = env.step(action)
        rews.append(rew)

        if done:
            break
    
    env.close()

    return np.sum(rews)


if __name__ == "__main__":
    '''
    TODO(Q1.2): Try filling in different numbers for params and run this script. Observe CartPole behavior
    '''

# =============================================================================
#     params = np.array([0.2, -0.2, 0.2, -0.01])
#     policy = CartPoleLinearPolicy(params)
#     rew = run_policy(policy, render=True)
# 
# =============================================================================
  #  print('Params {} got reward {}'.format(params, rew))

    '''
    TODO(Q1.3): Sample 1000 policies and run all:
    - Plot histogram of total rewards
    - Count what percentage of random policies achieved full reward (500)

    '''
    a = np.random.random(4)
    print(a)
    policies = [CartPoleLinearPolicy(a) for _ in range(1000)]
    
    #print(policies)
    all_rews = np.array([run_policy(policy, render=False) for policy in policies])
   # print(all_rews)
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_rews)
    plt.xlabel('Reward')
    plt.ylabel('Count')
    plt.title('CartPole Random Policy Performance Histogram')
    plt.savefig('p1_q1.png')
    plt.show()

    num_full_rewards = np.sum(all_rews == 500)
    #print(num_full_rewards)
    print('Percetange of policies that achieved full reward: {}'.format(num_full_rewards / len(all_rews)))
