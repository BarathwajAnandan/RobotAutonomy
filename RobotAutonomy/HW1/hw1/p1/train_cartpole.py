import numpy as np
import matplotlib.pyplot as plt

from run_cartpole import CartPoleLinearPolicy, run_policy
from finite_diff import scalar_finite_diff


def train_cartpole_policy(num_training_iterations, num_rollouts_per_eval, learning_rate):

    def eval_params(params):
        '''
        TODO(Q3.1): 

        Implement the eval_params function, which takes in a policy params and:
        - forms a policy
        - runs it for num_rollouts_per_eval number of times
        - returns the mean reward across these rollouts
        '''
        rew_list = []
        policy = CartPoleLinearPolicy(params)
        for i in range(num_rollouts_per_eval):
            
            rewards = run_policy(policy)
            rew_list.append(rewards)
        return np.mean(rew_list)
    
    
    
    params = np.zeros(4)
   

    rews = []
    
    h = np.ones(4) * 1e-1
    for _ in range(num_training_iterations):
        
        f = lambda params : eval_params(params) 
        
        
        
        Nabla = scalar_finite_diff(f,params,h)
 
        
       #print(Nabla)
        params+= np.multiply( Nabla,learning_rate) 
        rews.append(eval_params(params))
         
        '''
        TODO(Q3.2): 
            
        - Implement the gradient ascent step to update params using scalar_finiite_diff
        - At the end of each iteration, use eval_params to compute the reward of a policy with the current params and record it in rews
        '''
        

    policy = params

    return policy, rews


if __name__ == "__main__":
    # Don't change these!
    max_reward = 500
    num_training_iterations = 20
    num_rollouts_per_eval = 20
    '''
    TODO(Q3.3): 
        
    
    
    - Plot the rews during training vs. training iterations
    - Tune the learning_rate so the training converges to max reward (500) in 7 iterations or less.
    - Record the values of the final policy parameters
    '''
    
    g = np.linspace(1,20,20)
    
# =============================================================================
#     i = np.linspace(10**-6,10* 10**-7,6)
#     
#     for k,j in enumerate(i):
#         learning_rate = j 
#         policy, rews = train_cartpole_policy(num_training_iterations, num_rollouts_per_eval, learning_rate)
#         print()
#         print(rews)
#         #print (policy)
#         if 500 in rews:
#             print(k)
# =============================================================================
            #plt.plot(g,rews)
    learning_rate = 10* 10**-6
    policy, rews = train_cartpole_policy(num_training_iterations, num_rollouts_per_eval, learning_rate)
    print(rews)
    plt.plot(g,rews)
    
    