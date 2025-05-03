from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import os
import torch

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

        # Automatically create new directories
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

        if self.agent == 'ppo':
            model = PPO("MlpPolicy", env, tensorboard_log=self.tensorboard_log_dir, verbose=1, 
                        device=self.device, n_steps=1024, batch_size=64)
        elif self.agent == 'dqn':
            model = DQN("MlpPolicy", env, tensorboard_log=self.tensorboard_log_dir, verbose=1,
                        device=self.device, batch_size=64)
        else:
            model = None
            raise ValueError('check agent name parameter to Vehicle Agent')

        self.model = model

    def learn(self):
        self.set_up()

        time_stamp = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoint_callback = CheckpointCallback(save_freq=50,
                                                 save_path=self.checkpoint_log_dir,
                                                 name_prefix='vehicle_rl_%s_%s' % (self.agent, time_stamp))

        timestep_callback = TimestepPrintCallback(print_freq=500)  # every 500 steps print

        reset_num_timesteps = True if self.load_model_dir is None else False

        self.model.learn(
            total_timesteps=self.total_time_steps,
            callback=[checkpoint_callback, timestep_callback],
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