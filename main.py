import os
import gym
import time
import math
import torch
import random
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict
from utils import register_single_patient_env, DiscretizeActionWrapper, save_learning_metrics

from memory import ReplayMemory, PrioritizedReplayBuffer, nstepReplayMemory
from agent import DQN, EFFDQN, Multistep

import warnings
from pkg_resources import PkgResourcesDeprecationWarning
warnings.filterwarnings("ignore", category=PkgResourcesDeprecationWarning)

# Save Path
EXPERIMENTS = "../Experiments/main/multistep_per/"

def train(env, args):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.device)

    metrics = defaultdict(list)

    # Set seeds
    seed = args.seed
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Initialize Replay Buffer
    if args.PER:
        replay_buffer = PrioritizedReplayBuffer(args, args.replay_buffer_size)
    else:
        replay_buffer = nstepReplayMemory(args, args.replay_buffer_size)

    # Initialize Agent
    if args.Agent == 'DQN':
        agent = DQN(args, env)
    elif args.Agent == 'EFFDQN':
        agent = EFFDQN(args, env)
    elif args.Agent == 'Multistep':
        agent = Multistep(args, env)

    checkpoint_freq = 1000

    # Training Loop
    agent.train()
    for i_episode in range(args.M_episodes):
        # Adaptive epsilon scheme
        eps = args.eps_end + (args.eps_start - args.eps_end) * math.exp((-1 * i_episode) / args.eps_decay)
        done = False
        current_return = 0
        state = env.reset()
        eff_action = env.min_action
        aug_state = torch.cat((torch.tensor([state[0]]).float().view(1, -1).to(args.device),
                               torch.tensor([eff_action]).float().view(1, -1).to(args.device)), dim=-1)

        while done == False:
            action = agent.select_action(aug_state, eps)  # act
            next_state, reward, done, info = env.step(np.array([action]))
            done_bool = float(done)

            current_return += reward  # 实际累积奖励直接累加，在选择动作时考虑λ衰减

            next_eff_action = args.lambdaa * eff_action + action

            next_aug_state = torch.cat((torch.tensor([next_state[0]]).float().view(1, -1).to(args.device),
                                        torch.tensor([next_eff_action]).float().view(1, -1).to(args.device)), dim=-1)

            replay_buffer.push(aug_state,
                               torch.LongTensor([action]),
                               torch.FloatTensor([reward]),
                               next_aug_state,
                               torch.tensor([1 - done_bool]))

            aug_state = next_aug_state
            eff_action = next_eff_action

            # Training
            if i_episode > args.EXPLORE and len(replay_buffer) > args.batch_size:
                if (i_episode+1) % args.replay_frequency == 0:
                    agent.learn(mem=replay_buffer)

            metrics['action_hist'].append(action)

        hyperglycemic_zone_len = np.where(np.array(env.env.env.BG_hist) > args.hyperglycemic_BG)[0].shape[0]
        hypoglycemic_zone_len = np.where(np.array(env.env.env.BG_hist) < args.hypoglycemic_BG)[0].shape[0]
        target_zone_len = len(env.env.env.BG_hist) - (hyperglycemic_zone_len + hypoglycemic_zone_len)

        # save
        metrics['training_reward'].append(round(current_return, 2))
        metrics['hyperglycemic_BG'].append(hyperglycemic_zone_len)
        metrics['hypoglycemic_BG'].append(hypoglycemic_zone_len)
        metrics['target_BG'].append(target_zone_len)

        metrics['BG_hist'].extend(env.env.env.BG_hist[:-1])
        metrics['CGM_hist'].extend(env.env.env.CGM_hist[:-1])
        metrics['insulin_hist'].extend(np.concatenate(env.env.env.insulin_hist).ravel().tolist())
        metrics['CHO_hist'].extend(env.env.env.CHO_hist)
        metrics['mortality'].append(env.env.env.BG_hist[-1])

        if i_episode % 1000 == 0:
            print(f"Episode: {i_episode + 1}  Reward: {current_return:.3f}  lr: {agent.optimizer.param_groups[0]['lr']}")

        if i_episode > args.EXPLORE and len(replay_buffer) > args.batch_size and args.scheduler:
            agent.scheduler.step()

        # if i_episode % checkpoint_freq == 0:
        #     save_learning_metrics(dir_, **metrics)
        #     agent.save(dir_)

    # Make Folder & Save
    expt_id = 'seed'+str(args.seed) + ' n'+str(args.multi_step)
    if not os.path.exists(EXPERIMENTS):
        os.makedirs(EXPERIMENTS)
    dir_ = (EXPERIMENTS + expt_id)
    os.makedirs(dir_)
    print(f"Created {dir_}")
    print('Network:', agent.policy_net, file=open(os.path.join(dir_, 'architecture.txt'), 'a'))
    save_learning_metrics(dir_, **metrics)
    agent.save(dir_)

    print('Done!')

    env.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="argument parser")
    parser.add_argument("--seed", type=int, default=3, help="Seed value (default: 3)")
    # env
    parser.add_argument("--patient_name", type=str, default='adult#009', help="Name of the patient")
    parser.add_argument("--hyperglycemic_BG", type=int, default=150)
    parser.add_argument("--hypoglycemic_BG", type=int, default=100)
    # reward
    parser.add_argument("--reward", type=str, default='zone_reward', help="Reward type (default: zone_reward)")
    parser.add_argument('--gamma', type=float, default=0.999, metavar='γ', help='discount')
    # Agent & networks
    parser.add_argument('--Agent', type=str)
    parser.add_argument("--state_embedding_size", type=int, default=16)
    parser.add_argument("--action_embedding_size", type=int, default=16)
    parser.add_argument("--n_hidden", type=int, default=256, help="hidden-size:Network hidden size")
    parser.add_argument("--num_hidden", type=int, default=2)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument('--drop_prob', type=float, default=0)
    # training settings
    parser.add_argument("--M_episodes", type=int, default=10000)
    parser.add_argument("--EXPLORE", type=int, default=1000, help='learn-start:Number of steps before starting training')
    parser.add_argument("--replay_buffer_size", type=int, default=int(1e6), help="memory-capacity")
    parser.add_argument('--replay_frequency', type=int, default=1, metavar='k', help='Learning frequency of policy_net')
    parser.add_argument('--TARGET_UPDATE', type=int, default=1, metavar='τ', help='target-update:Number of steps after which to update target network')
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='η', help='Learning rate')
    # ad-hoc
    parser.add_argument('--lambdaa', type=float, default=0.95, metavar='λ', help='exponential decay assumption on the actions')
    # epsilon scheme
    parser.add_argument('--eps_start', type=float, default=0.9)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument("--eps_decay", type=int, default=500)
    # lr_scheduler
    parser.add_argument('--scheduler', action='store_true', help='Inputting command means using lr_scheduler')
    parser.add_argument("--lr_decay", type=float, default=0.999, help="lr_decay_rate (default: 0.999)")
    # PER
    parser.add_argument('--PER', action='store_true', help='Inputting command means using PER')
    parser.add_argument('--priority_exponent', type=float, default=0.5, metavar='ω', help='Prioritised experience replay exponent (originally denoted α)')
    parser.add_argument('--priority_weight', type=float, default=0.4, metavar='β', help='Initial prioritised experience replay importance sampling weight')
    # multistep
    parser.add_argument('--history_length', type=int, default=1, metavar='T',
                        help='Number of consecutive states processed')
    parser.add_argument('--multi_step', type=int, default=1, metavar='n', help='Number of steps for multi-step return')
    # split
    parser.add_argument('--split', action='store_true', help='Inputting command means using split')
    # run
    parser.add_argument("--rb", type=int, default=1)
    parser.add_argument("--re", type=int, default=5)
    parser.add_argument("--ns", type=int, nargs='+', help='an integer for the list')

    args = parser.parse_args()
    args.priority_weight_increase = (1 - args.priority_weight) / (args.M_episodes - args.EXPLORE)

    print('Training for patient: ', args.patient_name)
    # Environment
    seed = 10
    env_id = register_single_patient_env(args.patient_name,
                                         reward_fun=args.reward,
                                         seed=seed,
                                         version='-v0')
    env = gym.make(env_id)
    env = DiscretizeActionWrapper(env, low=0, high=5, n_bins=6)

    # args.Agent = 'Multistep'
    # args.PER = True
    # args.rb = 3
    # args.re = 4
    # args.batch_size = 256

    for run in range(args.rb,args.re):
        args.seed = run + 1
        for n in args.ns:
            args.multi_step = n

            # Train
            start_time = time.time()
            train(env, args)
            end_time = time.time()
            print('Total Time:', (end_time - start_time) / 60, 'mins for', args.M_episodes, 'episodes.')
            time_record = open(EXPERIMENTS + 'time.txt', 'a')
            print('seed {} n {} time {}'.format(args.seed, n, (end_time - start_time) / 60), file=time_record)
            time_record.close()
