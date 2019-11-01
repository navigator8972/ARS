"""

Code to load a policy and generate rollout data. Adapted from https://github.com/berkeleydeeprlcourse. 
Example usage:
    python run_policy.py ../trained_policies/Humanoid-v1/policy_reward_11600/lin_policy_plus.npz Humanoid-v1 --render \
            --num_rollouts 20
"""
import numpy as np
import gym

from character_data_utils import collectCharacterPointBlob
import datetime

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert rollouts')
    args = parser.parse_args()

    print('loading and building expert policy')
    lin_policy = np.load(args.expert_policy_file, allow_pickle=True)

    lin_policy = list(lin_policy.items())[0][1]
    
    M = lin_policy[0]
    # mean and std of state vectors estimated online by ARS. 
    mean = lin_policy[1]
    std = lin_policy[2]
        
    env = gym.make(args.envname)

    returns = []
    observations = []
    actions = []

    timestep_limit = 200

    point_clouds = []

    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0

        rollout_point_clouds = []
        while not done:
            action = np.dot(M, (obs - mean)/std)
            observations.append(obs)
            actions.append(action)
            
            
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, timestep_limit))
            # if steps >= env.spec.timestep_limit:
            if steps >= timestep_limit:
            
                break

            point_frame = collectCharacterPointBlob(env)
            if point_frame is not None:
                rollout_point_clouds.append(point_frame)

        returns.append(totalr)
        print('Collected {0} frames'.format(len(rollout_point_clouds)))
        if rollout_point_clouds:
            point_clouds.append(np.array(rollout_point_clouds))
            print(np.array(rollout_point_clouds).shape)

    np.savez_compressed(args.envname+'-{date:%Y-%m-%d_%H:%M:%S}'.format(date=datetime.datetime.now()), data=point_clouds)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    
if __name__ == '__main__':
    main()
