import numpy as np
import torch
import torch.nn as nn
import gym
import time
import os
import inspect
import logz
from model import PolicyNetwork, ValueNetwork
from torch.optim import Adam
from multiprocessing import Process

def policy_gradient_loss(log_prob, adv, num_path):
    return - (log_prob.view(-1, 1) * adv).sum() / num_path

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
        return PolicyNetwork(input_size, [size for _ in range(n_layers)], output_size,
                             [activation() for _ in range(n_layers)], output_activation, discrete, tanh_mean, tanh_std)

    elif network == 'value':
        return ValueNetwork(input_size, [size for _ in range(n_layers)], 1,
                            [activation() for _ in range(n_layers)], output_activation)

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
                ac_, log_prob, _ = policy(ob)
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
        # if itr % 10 == 0:
        #     print('weights in layer1')
        #     print(list(policy.parameters())[0].grad)
        #     print('bias in layer1')
        #     print(list(policy.parameters())[1].grad)
        #     print('weights in layer2')
        #     print(list(policy.parameters())[2].grad)
        #     print('bias in layer2')
        #     print(list(policy.parameters())[3].grad)

        policy_optimizer.step()

        # Log diagnostics
        returns = [path["reward"].sum() for path in paths]
        returns_all.append(sum(returns)/len(returns))

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
    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--n_iter', '-n', type=int, default=3)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-2)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
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

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    np.savetxt(args.exp_name, np.array(returns_all_list))


    # sns.tsplot(data=np.array(returns_all_list), color='blue')
    # plt.xlabel('Epoch')
    # plt.ylabel('Return')
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    main()













