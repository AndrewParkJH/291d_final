from RL.agents.vehicle_agent import VehicleAgent
from RL.environment.multi_vehicle_env import MultiVehicleEnv

sim_kwargs = {
    'trip_date': "2019-09-17",
    'simulation_start_time': 7*3600,
    'simulation_end_time': 10*3600,
    'accumulation_time': 120,
    'num_vehicles': 4,
    'randomize_vehicle_position': True,
    'vehicle_capacity': 10
}

def main():
    agent = VehicleAgent(MultiVehicleEnv, sim_kwargs, agent_name='ppo', total_time_steps=10000000)
    agent.learn()

if __name__ == "__main__":
    main()
