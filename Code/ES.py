import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import gym
import gym.spaces
import numpy as np
import argparse
import os
from torch import Tensor
from torch.distributions import Categorical
from torch.distributions import Normal
import time
## this is the version of ES where we use traces from theta to estimate fitness of theta_i
## the current working code on ES with importance sampling
parser = argparse.ArgumentParser()
parser.add_argument('--envname', type=str, default='HalfCheetah-v2', help='name of the gym env')
# parser.add_argument("--max_timesteps", type=int, default=)
parser.add_argument('-n','--num_train_epoch', type=int, default=100,
                    help='Number of training epoches')
parser.add_argument('-ep','--episode_length', type=int, default=100,
                    help='Number of max episode length')
parser.add_argument('--random_seed', type=int, default=42,
                    help='random seed')
parser.add_argument('-pop','--pop_size', type=int, default=50,
                    help='size of ES population')
parser.add_argument('-e','--n_experiment', type=int, default=1,
                    help='Number of experiment to run')
parser.add_argument('-lr','--learning_rate', type=float, default=0.5,
                    help='model learning rate')
parser.add_argument('-sigma','--sigma', type=float, default=0.05,
                    help='standard deviation of ES noise')
parser.add_argument('-hn','--n_hidden_neuron', type=int, default=32,
                    help='Number of hidden layer neurons')
parser.add_argument('--mirror_sampling', '-ms', action='store_false',
                    help='set flag to use mirror sampling')
parser.add_argument('--fitness_shaping', '-fs', action='store_false',
                    help='set flag to use fitness shaping')
parser.add_argument('--weight_decay', '-wd', type=float, default=1.0,
                    help='percent of weight to keep, default 1.0')
parser.add_argument('--action_sigma', type=float, default=0.05,
                    help='the sigma used in generating continuous actions')
parser.add_argument('--exp_name', type=str, default='trial_run.txt', help='specify name of log file')
# parser.add_argument('--hidden_neuron', type=int, default=32,
#                     help='Number of hidden layer neurons')
args = parser.parse_args()

DEBUG = True
starttime = time.time()

if DEBUG:
    args.episode_length = 100
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!IN DEBUG MODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!IN DEBUG MODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!IN DEBUG MODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

print(args)
sys.stdout.flush()

class ES_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, is_discrete, action_sigma):
        ## save the parameters for copying model
        super(ES_Model, self).__init__()
        self.input = nn.Linear(input_dim,hidden_dim,bias=False)
        self.output = nn.Linear(hidden_dim,output_dim,bias=False)

        self.is_discrete = is_discrete

        ## In this implementation we just set weights to init as zero
        ## don't worry about bias to make things simpler
        self.input.weight.data.zero_()
        self.output.weight.data.zero_()

        ## these are the epsilon for theta_i
        self.epsilon_input = None
        self.epsilon_output = None

        self.action_sigma = action_sigma

    def forward(self, x):
        out = F.relu(self.input(x))
        out = self.output(out)
        if self.is_discrete: ## if discrete then give action probabilities
            log_probs = F.log_softmax(out, dim=1)
            return log_probs
        else: ## if continuous then give each action as value between -1 and 1
            return F.tanh(out)

    def get_action(self, x):
        out = self.forward(x)
        if self.is_discrete:
            log_probs = out
            dist = Categorical(logits=log_probs)
            action = dist.sample()
            action_prob = dist.log_prob(action).exp()
            env_action = action.item()
            return env_action, [out, dist, action_prob]
        else:
            dist = Normal(out,Tensor([self.action_sigma]))
            action = dist.sample()
            env_action = action.data.numpy().reshape(-1)
            action_prob = dist.log_prob(action).exp()
            return env_action, [out, dist, action_prob]
    def get_parameter_distance(self, other):
        ## get the l2 distance in parameter, given another model
        distance = 0
        distance += torch.sum((self.input.weight.data - other.input.weight.data)**2)
        distance += torch.sum((self.output.weight.data - other.output.weight.data)**2)
        distance = distance.detach().numpy()
        return np.sqrt(distance)

    def weight_decay(self, decay_rate):
        self.input.weight.data *= decay_rate
        self.output.weight.data *= decay_rate

    def mutate(self, sigma, return_epsilon=False):
        ## call this for theta_i, after sync with theta_center
        ## to add epsilon
        ## return list of epsilon
        input_shape = self.input.weight.data.shape
        self.epsilon_input = Normal(torch.zeros(input_shape),sigma).sample()
        output_shape = self.output.weight.data.shape
        self.epsilon_output = Normal(torch.zeros(output_shape),sigma).sample()
        self.input.weight.data += self.epsilon_input
        self.output.weight.data += self.epsilon_output
        if return_epsilon:
            return [self.epsilon_input.clone(), self.epsilon_output.clone()]

    def mutate_with_negative_epsilon(self, eps_list):
        ## given list of epsilon, use the negative of those epsilon to update its weights
        ## used in mirror sampling
        self.epsilon_input = -eps_list[0]
        self.epsilon_output = -eps_list[1]
        self.input.weight.data += self.epsilon_input
        self.output.weight.data += self.epsilon_output

    def update_theta_center(self,theta_i, alpha, fitness, pop_size, sigma):
        ## update theta center with a theta_i
        self.input.weight.data += alpha * fitness * theta_i.epsilon_input / (pop_size*sigma)
        self.output.weight.data += alpha * fitness * theta_i.epsilon_output / (pop_size*sigma)

class Net_Baseline(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Net_Baseline, self).__init__()
        self.input = nn.Linear(input_dim,hidden_dim)
        self.hidden = nn.Linear(hidden_dim,hidden_dim)
        self.output = nn.Linear(hidden_dim,1)
        self.layers = [self.input,self.hidden,self.output]
        self.init_weights()
    def forward(self, x):
        out = F.relu(self.input(x))
        out = F.relu(self.hidden(out))
        out =  F.tanh(self.output(out))
        return out
    def init_weights(self): ##He et at. init
        for layer in self.layers:
            n_in = layer.weight.data.shape[1]
            layer.weight.data.normal_(0.0,2.0/n_in)

def compute_fitness(env, model, episode_length, n_episode=1):
    fitness = 0
    data_count = 0
    for i_episode in range(n_episode):
        observation = env.reset()
        for t in range(episode_length):
            data_count+=1
            # env.render()

            env_action, _ = model.get_action(Tensor(observation).reshape(1, -1))

            ## take action in env
            observation, reward, done, info = env.step(env_action)
            fitness += reward
            if done:
                break
        ## now we have finished one episode, we now assign reward (all the data points in
        ## the same trajectory have the same reward)
    return fitness/n_episode, data_count

def get_new_population(theta_center, theta_list, sigma, mirror_sampling):
    ## in place operation to generate new population in theta_list
    pop_size = len(theta_list)
    if not mirror_sampling:
        for i in range(pop_size):
            theta_i = theta_list[i]
            theta_i.load_state_dict(theta_center.state_dict())  # first sync with theta_center
            theta_i.mutate(sigma)
    else:## mirror sampling
        assert pop_size % 2 == 0
        half_pop_size = int(pop_size/2)
        for i in range(half_pop_size):
            theta_i = theta_list[i]
            theta_i.load_state_dict(theta_center.state_dict())  # first sync with theta_center
            eps_list = theta_i.mutate(sigma,return_epsilon=True)
            theta_j = theta_list[half_pop_size+i]
            theta_j.load_state_dict(theta_center.state_dict())
            theta_j.mutate_with_negative_epsilon(eps_list)

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks
def compute_centered_ranks(x):
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= (x.size - 1)
    y -= .5
    return y
def get_reshaped_fitness(fitness_list):
    ## given the fitness list, return a reshaped fitness list
    return compute_centered_ranks(np.array(fitness_list))
def get_reward_to_go(rewards, gamma):
    ## given rewards in one episode, calculate reward to go
    ## return a list of reward to go (q_n)
    ## faster implementation
    T = len(rewards)
    q = 0
    q_n = []
    for t in range(T-1,-1,-1):
        q = rewards[t] + gamma*q
        q_n.append(q)
    q_n.reverse()
    return q_n

def train_ES(envname,
             num_train_epoch,
             random_seed,
             learning_rate,
             episode_length,
             sigma,
             pop_size,
             n_hidden,
             mirror_sampling,
             fitness_shaping,
             weight_decay_rate,
             action_sigma
             ):
    ## make environment
    env = gym.make(envname)
    ## set episode length
    episode_length = episode_length or env.spec.max_episode_steps

    ###### get env info
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    n_observation = env.observation_space.shape[0]
    if is_discrete:
        n_action = env.action_space.n
    else:
        n_action = env.action_space.shape[0]

    if DEBUG:
        print(env.action_space.high)
        print(env.action_space.low)
        print(n_action)
        print(n_observation)
        print(env.spec.max_episode_steps)

    ## seed env and libraries
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    ## init theta center and theta_i as empty models
    theta_center = ES_Model(n_observation,n_hidden,n_action,is_discrete, action_sigma)
    theta_list = [ES_Model(n_observation,n_hidden,n_action,is_discrete, action_sigma) for _ in range(pop_size)]

    fitness_list = np.zeros(pop_size) ## the fitness of each theta_i
    center_fitness_list = [] ## this is for log and plot

    naive_es_data_used = 0
    for i_gen in range(num_train_epoch):
        ## weight decay for center
        if weight_decay_rate<1:
            theta_center.weight_decay(weight_decay_rate)

        ## first generate theta_i population
        get_new_population(theta_center, theta_list, sigma, mirror_sampling)

        for i in range(pop_size):
            fitness, data_count = compute_fitness(env, theta_list[i], episode_length)
            naive_es_data_used += data_count
            fitness_list[i] = fitness

        if fitness_shaping:
            fitness_list = get_reshaped_fitness(fitness_list)

        ## get mean and std of fitness for normalization
        ave_fit = np.mean(fitness_list)
        fit_std = np.std(fitness_list) + 1e-2

        for i_pop in range(pop_size):
            ## normalize fitness
            fitness = fitness_list[i_pop]
            fitness -= ave_fit
            fitness /= fit_std
            theta_i = theta_list[i_pop]

            ## now update theta_center
            theta_center.update_theta_center(theta_i, learning_rate, fitness, pop_size, sigma)

        center_return, _ = compute_fitness(env, theta_center,episode_length, 10)
        print('gen', i_gen, 'center', center_return, 'ave', ave_fit)
        center_fitness_list.append(center_return)
    print('naive ES total data usage:',naive_es_data_used)

    return center_fitness_list

center_fitness_all = []
n_experiment = args.n_experiment
for i in range(n_experiment):
    random_seed = (i+1) * args.random_seed
    center_fitness_list = train_ES(envname=args.envname,
                                   num_train_epoch=args.num_train_epoch,
                                   random_seed=random_seed,
                                   learning_rate=args.learning_rate,
                                   episode_length=args.episode_length,
                                   sigma=args.sigma,
                                   pop_size=args.pop_size,
                                   n_hidden=args.n_hidden_neuron, mirror_sampling=args.mirror_sampling
                                   , fitness_shaping=args.fitness_shaping,
                                   weight_decay_rate=args.weight_decay,
                                   action_sigma=args.action_sigma)

    center_fitness_all.append(center_fitness_list)

if DEBUG:
    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = sns.tsplot(data=np.array(center_fitness_all), color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Return')
    plt.tight_layout()
    plt.show()

# training_log_all_experiment = np.array(center_fitness_all)
# save_path = os.path.join('logs',args.exp_name)
# np.savetxt(save_path,training_log_all_experiment)
    


















