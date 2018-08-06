import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import time
import os
import inspect
import logz
from model import PolicyNetwork, ValueNetwork
from torch.optim import Adam
from multiprocessing import Process
from torch.distributions import Categorical
from torch.distributions import Normal


class PGES_model(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activations, action_sigma=0.05, out_activation=None, is_discrete=True,
                 tanh_mean=False, tanh_std=False):
        super(PGES_model, self).__init__()

        self.is_discrete = is_discrete

        fc = [nn.Linear(input_dim, hidden_dims[0]), activations[0]]
        for i in range(len(hidden_dims) - 1):
            fc.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            fc.append(activations[i + 1])

        if self.is_discrete:
            fc.append(nn.Linear(hidden_dims[-1], output_dim))
        else:
            self.output_dim = output_dim
            self.mean = nn.Linear(hidden_dims[-1], output_dim)
            self.log_std = nn.Linear(hidden_dims[-1], output_dim)
            self.tanh_mean = tanh_mean
            self.tanh_std = tanh_std

        if out_activation is not None:
            fc.append(out_activation())

        self.fc_list = fc
        self.fc = nn.Sequential(*fc)

        ## these are the epsilon for theta_i
        self.epsilon_input = None
        self.epsilon_output = None

        self.action_sigma = action_sigma

    def forward(self, o):
        if self.is_discrete:
            logits = self.fc(o)
            probs = F.softmax(logits, dim=1)
            log_probs = F.log_softmax(logits, dim=1)
            distr = Categorical(probs)
            sampled_actions = distr.sample()
            sampled_log_prob = distr.log_prob(sampled_actions)
            return sampled_actions, sampled_log_prob, log_probs

        else:
            representation = self.fc(o)
            if self.tanh_mean:
                mean = F.tanh(self.mean(representation))
            else:
                mean = self.mean(representation)
            if self.tanh_std:
                log_std = F.tanh(self.log_std(representation))
            else:
                log_std = self.log_std(representation)

            distr = Normal(loc=mean, scale=log_std.exp())
            sampled_actions = distr.rsample().detach()
            temp = distr.log_prob(sampled_actions)
            sampled_log_prob = temp.sum(1).view(-1, 1)

            return sampled_actions, mean, sampled_log_prob, distr

    def get_action(self, x):
        if self.is_discrete:
            _, _, out = self.forward(x)
            log_probs = out
            dist = Categorical(logits=log_probs)
            action = dist.sample()
            action_prob = dist.log_prob(action).exp()
            env_action = action.item()
            return env_action, [out, dist, action_prob]
        else:
            action, out, _, dist = self.forward(x)
            env_action = action.data.numpy().reshape(-1)
            action_prob = dist.log_prob(action).exp()
            return env_action, [out, dist, action_prob]

    def get_parameter_distance(self, other):
        ## get the l2 distance in parameter, given another model
        distance = 0
        distance += torch.sum((self.input.weight.data - other.input.weight.data) ** 2)
        distance += torch.sum((self.output.weight.data - other.output.weight.data) ** 2)
        distance = distance.detach().numpy()
        return np.sqrt(distance)

    def weight_decay(self, decay_rate):
        self.input.weight.data *= decay_rate
        self.output.weight.data *= decay_rate

    def mutate(self, sigma, return_epsilon=False):
        ## call this for theta_i, after sync with theta_center
        ## to add epsilon
        ## return list of epsilon
        input_shape = self.fc_list[0].weight.data.shape
        self.epsilon_input = Normal(torch.zeros(input_shape), sigma).sample()
        try:
            output_shape = self.fc_list[-2].weight.data.shape
        except AttributeError:
            output_shape = self.fc_list[-1].weight.data.shape
        self.epsilon_output = Normal(torch.zeros(output_shape), sigma).sample()
        self.fc_list[0].weight.data += self.epsilon_input
        try:
            self.fc_list[-2].weight.data += self.epsilon_output
        except AttributeError:
            self.fc_list[-1].weight.data += self.epsilon_output
        if return_epsilon:
            return [self.epsilon_input.clone(), self.epsilon_output.clone()]

    def mutate_with_negative_epsilon(self, eps_list):
        ## given list of epsilon, use the negative of those epsilon to update its weights
        ## used in mirror sampling
        self.epsilon_input = -eps_list[0]
        self.epsilon_output = -eps_list[1]
        self.fc_list[0].weight.data += self.epsilon_input
        try:
            self.fc_list[-2].weight.data += self.epsilon_output
        except AttributeError:
            self.fc_list[-1].weight.data += self.epsilon_output

    def update_theta_center(self, theta_i, alpha, fitness, pop_size, sigma):
        ## update theta center with a theta_i
        self.fc_list[0].weight.data += alpha * fitness * theta_i.epsilon_input / (pop_size * sigma)
        try:
            self.fc_list[-2].weight.data += alpha * fitness * theta_i.epsilon_output / (pop_size * sigma)
        except AttributeError:
            self.fc_list[-1].weight.data += alpha * fitness * theta_i.epsilon_output / (pop_size * sigma)

def compute_fitness(env, model, episode_length, n_episode=1):
    fitness = 0
    data_count = 0
    # print(episode_length)
    for i_episode in range(n_episode):
        observation = env.reset()
        for t in range(episode_length):
            data_count += 1
            # env.render()

            env_action, _ = model.get_action(torch.Tensor(observation).reshape(1, -1))

            ## take action in env
            observation, reward, done, info = env.step(env_action)
            fitness += reward
            if done:
                break
        ## now we have finished one episode, we now assign reward (all the data points in
        ## the same trajectory have the same reward)
    return fitness / n_episode, data_count


def get_new_population(theta_center, theta_list, sigma, mirror_sampling):
    ## in place operation to generate new population in theta_list
    pop_size = len(theta_list)
    if not mirror_sampling:
        for i in range(pop_size):
            theta_i = theta_list[i]
            theta_i.load_state_dict(theta_center.state_dict())  # first sync with theta_center
            theta_i.mutate(sigma)
    else:  ## mirror sampling
        assert pop_size % 2 == 0
        half_pop_size = int(pop_size / 2)
        for i in range(half_pop_size):
            theta_i = theta_list[i]
            theta_i.load_state_dict(theta_center.state_dict())  # first sync with theta_center
            eps_list = theta_i.mutate(sigma, return_epsilon=True)
            theta_j = theta_list[half_pop_size + i]
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
    for t in range(T - 1, -1, -1):
        q = rewards[t] + gamma * q
        q_n.append(q)
    q_n.reverse()
    return q_n

def build_network(
        input_size,
        output_size,
        discrete=True,
        network='policy',
        n_layers=1,
        size=32,
        activation=nn.Tanh,
        output_activation=None,
        tanh_mean=False,
        tanh_std=False
        ):
    if network == 'policy':
        return PGES_model(input_size, [size for _ in range(n_layers)], output_size,
                             [activation() for _ in range(n_layers)], out_activation=output_activation, is_discrete=discrete, tanh_mean=tanh_mean, tanh_std=tanh_std)

    elif network == 'value':
        return ValueNetwork(input_size, [size for _ in range(n_layers)], 1,
                            [activation() for _ in range(n_layers)], output_activation)

### ES code starts here !!!!

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
             network):
    ## make environment
    env = gym.make(envname)
    ## set episode length
    episode_length = episode_length or env.spec.max_episode_steps

    ###### get env info
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    n_observation = env.observation_space.shape[0]
    if is_discrete:
        n_action = env.action_space.n
        activations = [nn.ReLU()]
    else:
        n_action = env.action_space.shape[0]
        activations = [nn.Tanh()]

    # if DEBUG:
    #     print(env.action_space.high)
    #     print(env.action_space.low)
    #     print(n_action)
    #     print(n_observation)
    #     print(env.spec.max_episode_steps)

    ## seed env and libraries
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    ## init theta center and theta_i as empty models
    theta_center = network

    theta_list = [PGES_model(n_observation, n_hidden, n_action, activations, action_sigma=0.05, is_discrete=is_discrete) for _ in range(pop_size)]

    fitness_list = np.zeros(pop_size)  ## the fitness of each theta_i
    center_fitness_list = []  ## this is for log and plot

    naive_es_data_used = 0
    for i_gen in range(num_train_epoch):
        ## weight decay for center
        if weight_decay_rate < 1:
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
        # print('theta center weights:')
        # theta_center.print_weight()
        # print('************')
        # print('new thetas weights:')
        # for each in theta_list:
        #     each.print_weight()

        center_return, _ = compute_fitness(env, theta_center, episode_length, 10)
        print('gen', i_gen, 'center', center_return, 'ave', ave_fit)
        center_fitness_list.append(center_return)
    print('naive ES total data usage:', naive_es_data_used)

    return theta_center.state_dict(), center_fitness_list

### ES code ends here !!!!!


def policy_gradient_loss(log_prob, adv, num_path):
    return - (log_prob.view(-1, 1) * adv).sum() / num_path

# input_dim, hidden_dims, output_dim, activations, action_sigma=0.05, out_activation=None, is_discrete=True,
#                  tanh_mean=False, tanh_std=False


def pathlength(path):
    return len(path['reward'])


def train_PG(exp_name='',
             env_name='CartPole-v0',
             n_iter=100,
             gamma=1.0,
             min_timesteps_per_batch=1000,
             max_path_length=None,
             learning_rate=5e-3,
             reward_to_go=True,
             animate=True,
             logdir=None,
             normalize_advantages=True,
             nn_baseline=False,
             seed=0,
             n_layers=2,
             size=64,
             tanh_mean=False,
             tanh_std=False,
             int_activation='relu'):
    start = time.time()

    # configure output directory for logging
    logz.configure_output_dir(logdir)

    # log experimental parameters
    args = inspect.signature(train_PG).parameters.keys()
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # make the gym environment
    env = gym.make(env_name)

    # Is the environment discrete or continuous
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    # observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    if int_activation == 'relu':
        activation = nn.ReLU
    elif int_activation == 'tanh':
        activation = nn.Tanh

    if discrete:
        policy = build_network(input_size=ob_dim, output_size=ac_dim, discrete=discrete, network='policy',
                               n_layers=n_layers, size=size, activation=activation)
    else:
        policy = build_network(input_size=ob_dim, output_size=ac_dim, discrete=discrete, network='policy',
                               n_layers=n_layers, size=size, tanh_mean=tanh_mean, tanh_std=tanh_std, activation=activation)

    policy_loss = policy_gradient_loss
    policy_optimizer = Adam(policy.parameters(), lr=learning_rate)

    if nn_baseline:
        baseline_prediction = build_network(input_size=ob_dim, output_size=1, network='value', n_layers=n_layers,
                                            size=size, activation=activation)
        baseline_loss = nn.MSELoss()
        baseline_optimizer = Adam(baseline_prediction.parameters(), lr=learning_rate)


    # Training loop
    returns_all = []

    total_timesteps = 0
    for itr in range(n_iter):
        print("******* Iteration %i ********"%itr)

        # collect paths until we have enough timesteps
        timesteps_this_batch = 0
        paths = []
        # new_loss = []
        while True:
            ob_ = env.reset()
            obs, acs, rewards, log_probs = [], [], [], []
            animate_this_episode = (len(paths) == 0 and (itr % 10 == 0) and animate)
            steps = 0
            while True:
                if animate_this_episode:
                    env.render()
                    time.sleep(0.05)
                ob = torch.from_numpy(ob_).float().unsqueeze(0)
                obs.append(ob)
                if discrete:
                    ac_, log_prob, _ = policy(ob)
                else:
                    ac_, _, log_prob, _ = policy(ob)
                acs.append(ac_)
                log_probs.append(log_prob)
                if discrete:
                    ac = int(ac_)
                else:
                    ac = ac_.squeeze(0).numpy()
                ob_, rew, done, _ = env.step(ac)
                # get the gradient
                # if itr % 5 == 0:
                #     print(-log_prob * rew)
                # new_loss.append(float(-log_prob * rew))
                # if (itr > 0) and (itr % 20 == 0):
                #     loss = float(-log_prob * rew)
                #     if loss < threshold:
                #         # do evolution strategy
                # stop
                rewards.append(rew)
                steps += 1
                if done or steps > max_path_length:
                    break
            path = {'observation': torch.cat(obs, 0),
                    'reward': torch.Tensor(rewards),
                    'action': torch.cat(acs, 0),
                    'log_prob': torch.cat(log_probs, 0)}
            paths.append(path)
            timesteps_this_batch += pathlength(path)
            if timesteps_this_batch > min_timesteps_per_batch:
                break
        total_timesteps += timesteps_this_batch

        ob_no = torch.cat([path['observation'] for path in paths], 0)
        ac_na = torch.cat([path['action'] for path in paths], 0)


        q_n = []
        for path in paths:
            rewards = path['reward']
            num_steps = pathlength(path)
            if reward_to_go:
                q_n.append(torch.cat([(torch.pow(gamma, torch.arange(num_steps - t)) * rewards[t:]).sum().view(-1, 1)
                                      for t in range(num_steps)]))
            else:
                q_n.append((torch.pow(gamma, torch.arange(num_steps)) * rewards).sum() * torch.ones(num_steps, 1))
        q_n = torch.cat(q_n, 0)

        if nn_baseline:
            b_n = baseline_prediction(ob_no)
            q_n_std = q_n.std()
            q_n_mean = q_n.mean()
            b_n_scaled = b_n * q_n_std + q_n_mean
            adv_n = (q_n - b_n_scaled).detach()
        else:
            adv_n = q_n

        if normalize_advantages:
            adv_n = (adv_n - adv_n.mean()) / (adv_n.std() + np.finfo(np.float32).eps.item())

        if nn_baseline:
            target = (q_n - q_n_mean) / (q_n_std + np.finfo(np.float32).eps.item())
            baseline_optimizer.zero_grad()
            b_loss = baseline_loss(b_n, target)
            b_loss.backward()
            baseline_optimizer.step()

        log_probs = torch.cat([path['log_prob'] for path in paths], 0)
        policy_optimizer.zero_grad()
        loss = policy_loss(log_probs, adv_n, len(paths))
        loss.backward()

        # old_parameters = list(policy.parameters())[0].grad
        # # if itr % 20 == 0:
        # #     print(list(policy.parameters())[0].grad)
        # if torch.median(list(policy.parameters())[0].grad) < torch.median(old_parameters):
        #     print('!!!!!!!ES starts here!!!!!!')
        #     print(list(policy.parameters())[0].grad)
        #     # train_ES(envname=envname,
        #     #          num_train_epoch=20,
        #     #          random_seed=seed,
        #     #          learning_rate=learning_rate,
        #     #          episode_length=max_path_length,
        #     #          sigma=0.05,
        #     #          pop_size=50,
        #     #          n_hidden=size,
        #     #          mirror_sampling=True,
        #     #          fitness_shaping=False,
        #     #          weight_decay_rate=1.0,
        #     #          action_sigma=0.05)
        # else:
        #     old_parameters = list(policy.parameters())[0].grad
        policy_optimizer.step()

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        returns_all.append(np.mean(returns))
        if itr == 0:
            last_returns = np.mean(returns)
            last_improvement = 0
        else:
            improvement = np.mean(returns) - last_returns
            if improvement < last_improvement - 10:
                print('!!!!!!!ES starts here!!!!!!')
                center_fitness_all = []
                n_experiment = 1
                theta_center_dict, _ = train_ES(env_name,
                                               num_train_epoch=10,
                                               random_seed=42,
                                               learning_rate=0.5,
                                               episode_length=max_path_length,
                                               sigma=0.05,
                                               pop_size=50,
                                               n_hidden=[32], mirror_sampling=False,
                                               fitness_shaping=False,
                                               weight_decay_rate=1.0,
                                               network=policy)

                policy.load_state_dict(theta_center_dict)
                last_improvement = improvement
                last_returns = np.mean(returns)


            else:
                last_improvement = improvement
                last_returns = np.mean(returns)


        ep_lengths = [pathlength(path) for path in paths]
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", np.mean(returns))
        logz.log_tabular("StdReturn", np.std(returns))
        logz.log_tabular("MaxReturn", np.max(returns))
        logz.log_tabular("MinReturn", np.min(returns))
        logz.log_tabular("EpLenMean", np.mean(ep_lengths))
        logz.log_tabular("EpLenStd", np.std(ep_lengths))
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        # logz.log_tabular("MinLoss", min(new_loss))
        # logz.log_tabular("MaxLoss", max(new_loss))
        logz.dump_tabular()
        if nn_baseline:
            logz.save_checkpoint({
                'policy_state_dict': policy.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict(),
                'baseline_state_dict': baseline_prediction.state_dict(),
                'baseline_optimizer': baseline_optimizer.state_dict()
            })
        else:
            logz.save_checkpoint({
                'policy_state_dict': policy.state_dict(),
                'policy_optimizer': policy_optimizer.state_dict()
            })
    return returns_all


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CartPole-v0', help='name of the gym env')
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=int, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    parser.add_argument('--tanh_mean', '-tm', action='store_true')
    parser.add_argument('--tanh_std', '-ts', action='store_true')
    parser.add_argument('--internal_activation', '-ia', type=str, default='tanh')
    args = parser.parse_args()

    if not (os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = os.path.join('data', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    returns_all_list = []
    for e in range(args.n_experiments):
        seed = args.seed + 10 * e
        print('Running experiment with seed %d' % seed)

        def train_func():
            return train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir, '%d' % seed),
                normalize_advantages=not (args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline,
                seed=seed,
                n_layers=args.n_layers,
                size=args.size,
                tanh_mean=args.tanh_mean,
                tanh_std=args.tanh_std,
                int_activation=args.internal_activation
            )

        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        # p = Process(target=train_func, args=tuple())
        # p.start()
        # p.join()
        returns_all = train_func()
        returns_all_list.append(returns_all)

    # print(logdir)
    np.savetxt(args.exp_name, np.array(returns_all_list))
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    #
    # ax = sns.tsplot(data=np.array(returns_all_list), color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Return')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()

# if __name__ == "__main__":
#     model = PGES_model(input_dim=2, hidden_dims=[64], output_dim=4, activations=[nn.ReLU()], is_discrete=False)
#     tensor_1 = torch.Tensor([[1, 1], [1, 2]])
#     print(model.forward(tensor_1))
