from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import os
import torch
import re

BASE_LOG_DIR = './RL/training/logs/'
BASE_MODEL_DIR = './RL/training/models/'

# TENSORBOARD_LOG_DIR = './RL/training/logs/checkpoint_log/vehicle_tensorboard_log'
# CHECKPOINT_LOG_DIR = './RL/training/logs/checkpoint_log/vehicle_training_log'
# MODEL_DIR = './RL/training/models/vehicle_models'

class VehicleAgent:
    def __init__(self, env, sim_kwargs, agent_name, total_time_steps = 200000,
                 load_model_dir=None, tensorboard_log=None, n_cpu=4):

        self.env = env
        self.sim_kwargs = sim_kwargs
        # self.n_env = n_env
        self.total_time_steps = total_time_steps
        self.load_model_dir = load_model_dir
        self.tensorboard_log = tensorboard_log
        self.agent = agent_name
        self.model = None
        self.n_cpu = n_cpu  # Number of CPU threads to use

        # Force CPU usage
        self.device = torch.device("cpu")
        
        # Set number of threads for PyTorch
        torch.set_num_threads(self.n_cpu)
        print(f"Using {self.n_cpu} CPU threads")

        # Automatically create new directories
        if load_model_dir:
            # Extract PPO14 from path
            match = re.search(r'PPO\d+', load_model_dir)
            self.run_name = match.group(0) if match else self.get_next_run_name()
        else:
            self.run_name = self.get_next_run_name()

        self.tensorboard_log_dir = os.path.join(BASE_LOG_DIR, self.run_name, 'tensorboard_log')
        self.checkpoint_log_dir = os.path.join(BASE_LOG_DIR, self.run_name, 'checkpoint_log')
        self.model_dir = os.path.join(BASE_MODEL_DIR, self.run_name)

        self.check_directories()

    def get_next_run_name(self):
        """
        Finds the next available PPO run folder like PPO1, PPO2, etc.
        """
        base_name = f"{self.agent.upper()}"
        run_idx = 1

        while os.path.exists(os.path.join(BASE_LOG_DIR, f"{base_name}{run_idx}")):
            run_idx += 1

        return f"{base_name}{run_idx}"

    def set_up(self):
        # env = self.env(self.sim_kwargs)
        env = Monitor(self.env(self.sim_kwargs))

        if self.load_model_dir is not None:
            print(f"Loading model from {self.load_model_dir}")
            if self.agent == 'ppo':
                self.model = PPO.load(self.load_model_dir, env=env, tensorboard_log=self.tensorboard_log_dir,
                                      tb_log_name='PPO_1',
                                      device=self.device)
            elif self.agent == 'dqn':
                self.model = DQN.load(self.load_model_dir, env=env, tensorboard_log=self.tensorboard_log_dir,
                                      device=self.device)
            else:
                raise ValueError('Unsupported agent type.')
        else:
            if self.agent == 'ppo':
                self.model = PPO("MlpPolicy", env,
                            tensorboard_log=self.tensorboard_log_dir,
                            verbose=1,
                            device=self.device
                            # n_steps=2048,
                            # batch_size=64,
                            # n_epochs=4,
                            # gae_lambda = 0.95,
                            # clip_range = 0.2,
                            # ent_coef = 0.1
                            )
            elif self.agent == 'dqn':
                self.model = DQN("MlpPolicy", env, tensorboard_log=self.tensorboard_log_dir, verbose=1,
                            device=self.device, batch_size=64)
            else:
                self.model = None
                raise ValueError('check agent name parameter to Vehicle Agent')

    def learn(self, callback=None):
        self.set_up()

        time_stamp = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoint_callback = CheckpointCallback(save_freq=1000,
                                                 save_path=self.checkpoint_log_dir,
                                                 name_prefix='vehicle_rl_%s_%s' % (self.agent, time_stamp))

        timestep_callback = TimestepPrintCallback(print_freq=500)  # every 500 steps print
        
        # Create a list of callbacks
        callbacks = [checkpoint_callback, timestep_callback]
        
        # Add the custom callback if provided
        if callback is not None:
            if isinstance(callback, list):
                callbacks.extend(callback)
            else:
                callbacks.append(callback)

        reset_num_timesteps = True if self.load_model_dir is None else False

        self.model.learn(
            total_timesteps=self.total_time_steps,
            callback=callbacks,
            reset_num_timesteps=reset_num_timesteps
        )

        model_dir = os.path.join(self.model_dir, '%s_vehicle_%s' % (self.agent, time_stamp))
        self.model.save(model_dir)

    def custom_learn(self, num_episodes=100, rollout_len=128):
        import numpy as np
        from stable_baselines3.common.buffers import RolloutBuffer

        env = self.env(self.sim_kwargs)
        model = self.model

        buffer = RolloutBuffer(rollout_len, env.observation_space, env.action_space, device=self.device)

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

    def check_directories(self):
        for dir_path in [self.tensorboard_log_dir, self.checkpoint_log_dir, self.model_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)


from stable_baselines3.common.callbacks import BaseCallback

class TimestepPrintCallback(BaseCallback):
    def __init__(self, print_freq=1000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"[{self.num_timesteps}] timesteps completed")
        return True