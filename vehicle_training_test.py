from RL.agents.vehicle_agent import VehicleAgent
from RL.environment.multi_vehicle_env import MultiVehicleEnv
import os
import torch

# Set environment variables to force CPU usage and optimize threading
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA
torch.set_num_threads(40)  # Use 8 CPU threads, adjust as needed

sim_kwargs = {
    'trip_date': "2019-09-17",
    'simulation_start_time': 7*3600,
    'simulation_end_time': 10*3600,
    'accumulation_time': 120,
    'num_vehicles': 1,
    'randomize_vehicle_position': True,
    'vehicle_capacity': 10
}

def main():
    agent = VehicleAgent(
        MultiVehicleEnv, 
        sim_kwargs, 
        agent_name='ppo', 
        total_time_steps=10000000,
        n_cpu=40  # Set the number of CPU threads to use
    )
    agent.learn()

if __name__ == "__main__":
    main()
