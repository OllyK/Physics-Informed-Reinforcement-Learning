from ray.rllib.algorithms.simple_q import SimpleQConfig


"""
config for reproducing the same result as original code
"""
single_worker = SimpleQConfig()
single_worker.training(
    model={
        'no_final_linear': True,
        'vf_share_layers': False,
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        "type": "ReplayBuffer",
        # "type": "MultiAgentReplayBuffer", # when num_workers > 0
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    # dueling=False,
    lr=0.0001,  # changed from 0.001, mw 
    gamma=0.99,
    train_batch_size=512,
    tau=0.1,
).resources(
    num_gpus=1
).rollouts(
    horizon=128,
    num_rollout_workers=0, # important!! each accounts for process
    num_envs_per_worker=1, # each accounts for process
    rollout_fragment_length=2,
# ).exploration(
#     explore=True,
#     exploration_config={
#         "type": "EpsilonGreedy",
#         'initial_epsilon': 0.99,
#         'final_epsilon': 0.01,
#         'epsilon_timesteps': 100000,
#     }
).exploration(
    explore=True,
    exploration_config={
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.1,
        "epsilon_timesteps": 150000,  # decay slower
    }
)



"""
config for parallelizing the original code
"""
multiple_worker = SimpleQConfig()
multiple_worker.training(
    model={
        'no_final_linear': True,
        'vf_share_layers': False,
    },
    target_network_update_freq=2000,
    replay_buffer_config={
        "_enable_replay_buffer_api": True,
        #"type": "ReplayBuffer",
        "type": "MultiAgentReplayBuffer", # when num_workers > 0
        "learning_starts": 1000,
        "capacity": 100000,
        "replay_sequence_length": 1,
    },
    # dueling=False,
    lr=0.0001,
    gamma=0.99,
    train_batch_size=512,
    tau=0.1,
).resources(
    #num_gpus=2,
    
    # Reserve 1 GPU for the trainer/learner (your single RTX-5080).
    ## Give Ray a CPU scheduling budget (16 workers + trainer + headroom).
    num_gpus=1,
).rollouts(
    horizon=512,
    #num_rollout_workers=16,
    num_rollout_workers=8,
    num_envs_per_worker=1,
    rollout_fragment_length=2,
    
    ## ensure workers do not try to grab GPUs (RLlib will treat this as 0 per worker).
    ## Some RLlib versions accept num_gpus_per_worker here; adding the key is harmless.
    ##num_gpus_per_worker=0,
# ).exploration(
#     explore=True,
#     exploration_config={
#         "type": "EpsilonGreedy",
#         'initial_epsilon': 0.99,
#         'final_epsilon': 0.01,
#         'epsilon_timesteps': 100000,
#     }
).exploration(
    explore=True,
    exploration_config={
        "type": "EpsilonGreedy",
        'initial_epsilon': 0.99,
        'final_epsilon': 0.01,
        'epsilon_timesteps': 100000,
    }
)
