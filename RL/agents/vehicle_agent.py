from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from datetime import datetime
import os

TENSORBOARD_LOG_DIR = './RL/training/logs/checkpoint_log/vehicle_tensorboard_log'
CHECKPOINT_LOG_DIR = './RL/training/logs/checkpoint_log/vehicle_training_log'
MODEL_DIR = './RL/training/models/vehicle_models'

class VehicleAgent:
    def __init__(self, env, sim_kwargs, agent_name, n_env=40, total_time_steps = 1000000,
                 load_model_dir=None, tensorboard_log=None):

        self.env = env
        self.sim_kwargs = sim_kwargs
        self.n_env = n_env
        self.total_time_steps = total_time_steps
        self.load_model_dir = load_model_dir
        self.tensorboard_log = tensorboard_log
        self.agent = agent_name
        self.model = None

        self.check_directories()

    def set_up(self):
        env = self.env(self.sim_kwargs)

        if self.agent == 'ppo':
            model = PPO("MlpPolicy", env, tensorboard_log=TENSORBOARD_LOG_DIR, verbose=1)
        elif self.agent == 'dqn':
            model = DQN("MlpPolicy", env, tensorboard_log=TENSORBOARD_LOG_DIR, verbose=1)
        else:
            model = None
            raise ValueError('check agent name parameter to Vehicle Agent')

        self.model = model

    def learn(self):
        self.set_up()

        time_stamp = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=CHECKPOINT_LOG_DIR,
                                                 name_prefix='vehicle_rl_%s_%s' % (self.agent, time_stamp))

        reset_num_timesteps = True if self.load_model_dir is None else False
        self.model.learn(total_timesteps=self.total_time_steps, callback=checkpoint_callback,
                         reset_num_timesteps=reset_num_timesteps)

        model_dir = os.path.join(MODEL_DIR, '%s_vehicle_%s' % (self.agent, time_stamp))
        self.model.save(model_dir)


    def custom_learn(self, num_episodes=100, rollout_len=128):
        import numpy as np
        from stable_baselines3.common.buffers import RolloutBuffer

        env = self.env(self.sim_kwargs)
        model = self.model

        buffer = RolloutBuffer(rollout_len, env.observation_space, env.action_space, device=model.device)

        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0

            while not done:
                actions = []
                mask = []

                for i, vehicle in enumerate(env.sim.network.vehicles):
                    if vehicle.has_new_request():
                        obs_i = obs[i]
                        action_i, _ = model.predict(obs_i, deterministic=False)
                        actions.append(action_i)
                        mask.append(True)
                    else:
                        actions.append(None)
                        mask.append(False)

                # Step the environment
                next_obs, rewards, done, _, _ = env.step(actions)

                # Store valid transitions only
                for i in range(len(actions)):
                    if mask[i]:
                        buffer.add(obs[i], actions[i], rewards[i], next_obs[i], done, {})

                obs = next_obs
                episode_reward += np.mean(rewards)

            print(f"Episode {ep} reward: {episode_reward:.2f}")

            # After one episode, train using collected data
            model.train()
            buffer.reset()

    @staticmethod
    def check_directories():
        if not os.path.exists(TENSORBOARD_LOG_DIR):
            os.makedirs(TENSORBOARD_LOG_DIR)

        if not os.path.exists(CHECKPOINT_LOG_DIR):
            os.makedirs(CHECKPOINT_LOG_DIR)

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
