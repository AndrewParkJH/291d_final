import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from visualizations.live.visualization_server import start_visualization_server, socketio, training_data, vehicle_states, request_data, simulation_time
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)  # Insert at the beginning of the path

from RL.environment.multi_vehicle_env import MultiVehicleEnv as VehicleRoutingEnv

class VisualizationCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(VisualizationCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.step_count = 0
        self.current_episode_reward = 0
        self.last_log_time = 0
        self.log_interval = 10  # Log every 10 steps

    def _on_step(self) -> bool:
        # Get the current environment
        env = self.training_env.envs[0]
        
        # Debug logging for vehicle states
        if self.step_count % self.log_interval == 0:
            print(f"\n=== Step {self.step_count} Debug Info ===")
            print(f"Simulation Time: {env.sim.env.now}")
            print(f"Number of Vehicles: {len(env.sim.network.vehicles)}")
            
            # Log vehicle states
            for i, vehicle in enumerate(env.sim.network.vehicles):
                print(f"\nVehicle {i} State:")
                print(f"  Position: {vehicle.current_pos}")
                print(f"  Current Node: {vehicle.current_node}")
                print(f"  Next Node: {vehicle.next_node}")
                print(f"  Passengers: {vehicle.current_num_pax}/{vehicle.max_capacity}")
                print(f"  Trip Sequence: {vehicle.trip_sequence}")
                print(f"  Current Requests: {vehicle.current_requests}")
                print(f"  Traversal Process Alive: {vehicle.traversal_process is not None and vehicle.traversal_process.is_alive}")
        
        # Get vehicle states
        global vehicle_states
        vehicle_states.clear()
        for i, vehicle in enumerate(env.sim.network.vehicles):
            # Convert numpy arrays to lists for position
            position = vehicle.current_pos
            if isinstance(position, np.ndarray):
                position = position.tolist()
            
            # Validate position coordinates
            if not all(isinstance(coord, (int, float)) and not np.isnan(coord) for coord in position):
                print(f"Warning: Invalid position coordinates for vehicle {i}:")
                print(f"  Raw position: {vehicle.current_pos}")
                print(f"  Converted position: {position}")
                print(f"  Types: {[type(coord) for coord in position]}")
                print(f"  NaN check: {[np.isnan(coord) for coord in position]}")
                continue
            
            vehicle_states[i] = {
                'id': i,
                'position': position,
                'passengers': [f"Passenger-{j}" for j in range(vehicle.current_num_pax)],
                'capacity': int(vehicle.max_capacity),  # Convert to Python int
                'route': [{'node_id': stop['node_id']} for stop in vehicle.trip_sequence] if vehicle.trip_sequence else [],
                'requests': list(vehicle.current_requests.keys())
            }
        
        # Get request data
        global request_data
        request_data = []
        
        # Debug logging for requests
        if self.step_count % self.log_interval == 0:
            print("\nRequest Data:")
            print(f"Number of Pending Requests: {len(env.sim.current_request) if hasattr(env.sim, 'current_request') else 0}")
        
        # Get assigned requests from vehicles
        for vehicle in env.sim.network.vehicles:
            for request_id, request in vehicle.current_requests.items():
                try:
                    # Get coordinates from the network
                    origin_node = request.get('o_r')
                    dest_node = request.get('d_r')
                    
                    if origin_node and dest_node:
                        # Get coordinates from the network
                        origin_coords = env.sim.network.get_node_coordinate(origin_node)
                        dest_coords = env.sim.network.get_node_coordinate(dest_node)
                        
                        if origin_coords and dest_coords:
                            # Convert to float and validate
                            origin_lat = float(origin_coords[1])  # lat is second element
                            origin_lon = float(origin_coords[0])  # lon is first element
                            dest_lat = float(dest_coords[1])
                            dest_lon = float(dest_coords[0])
                            
                            # Validate coordinates
                            if not all(isinstance(coord, (int, float)) and not np.isnan(coord) 
                                      for coord in [origin_lat, origin_lon, dest_lat, dest_lon]):
                                print(f"Warning: Invalid coordinates for request {request_id}:")
                                print(f"  Raw origin_coords: {origin_coords}")
                                print(f"  Raw dest_coords: {dest_coords}")
                                print(f"  Processed coordinates: origin=({origin_lat}, {origin_lon}), dest=({dest_lat}, {dest_lon})")
                                continue
                            
                            request_data.append({
                                'id': request_id,
                                'origin_lat': origin_lat,
                                'origin_lon': origin_lon,
                                'destination_lat': dest_lat,
                                'destination_lon': dest_lon,
                                'request_time': float(request['request_time'] if 'request_time' in request else env.sim.env.now),
                                'num_passengers': int(request['num_passengers']),
                                'status': 'assigned'
                            })
                            
                            # Debug logging for request processing
                            if self.step_count % self.log_interval == 0:
                                print(f"Processed Assigned Request {request_id}:")
                                print(f"  Origin: ({origin_lat}, {origin_lon})")
                                print(f"  Destination: ({dest_lat}, {dest_lon})")
                                print(f"  Status: assigned")
                except (ValueError, TypeError, IndexError, KeyError) as e:
                    print(f"Error processing coordinates for request {request_id}:")
                    print(f"  Error: {str(e)}")
                    print(f"  Request data: {request}")
                    continue
        
        # Get pending requests from the environment
        if hasattr(env.sim, 'current_request') and env.sim.current_request:
            for request in env.sim.current_request:
                if isinstance(request, dict):
                    # Get coordinates from the network
                    origin_node = request.get('o_r')
                    dest_node = request.get('d_r')
                    
                    if origin_node and dest_node:
                        # Get coordinates from the network
                        origin_coords = env.sim.network.get_node_coordinate(origin_node)
                        dest_coords = env.sim.network.get_node_coordinate(dest_node)
                        
                        if origin_coords and dest_coords:
                            # Convert to float and validate
                            try:
                                origin_lat = float(origin_coords[1])  # lat is second element
                                origin_lon = float(origin_coords[0])  # lon is first element
                                dest_lat = float(dest_coords[1])
                                dest_lon = float(dest_coords[0])
                                
                                # Validate coordinates
                                if not all(isinstance(coord, (int, float)) and not np.isnan(coord) 
                                          for coord in [origin_lat, origin_lon, dest_lat, dest_lon]):
                                    print(f"Warning: Invalid coordinates for pending request:")
                                    print(f"  Raw origin_coords: {origin_coords}")
                                    print(f"  Raw dest_coords: {dest_coords}")
                                    print(f"  Processed coordinates: origin=({origin_lat}, {origin_lon}), dest=({dest_lat}, {dest_lon})")
                                    continue
                                
                                request_data.append({
                                    'id': request.get('id', f'pending_{len(request_data)}'),
                                    'origin_lat': origin_lat,
                                    'origin_lon': origin_lon,
                                    'destination_lat': dest_lat,
                                    'destination_lon': dest_lon,
                                    'request_time': float(request.get('t_r^r', env.sim.env.now)),
                                    'num_passengers': int(request.get('num_passengers', 1)),
                                    'status': 'pending'
                                })
                                
                                # Debug logging for pending request processing
                                if self.step_count % self.log_interval == 0:
                                    print(f"Processed Pending Request {request.get('id', 'unknown')}:")
                                    print(f"  Origin: ({origin_lat}, {origin_lon})")
                                    print(f"  Destination: ({dest_lat}, {dest_lon})")
                                    print(f"  Status: pending")
                            except (ValueError, TypeError, IndexError) as e:
                                print(f"Error processing coordinates for pending request:")
                                print(f"  Error: {str(e)}")
                                print(f"  Raw origin_coords: {origin_coords}")
                                print(f"  Raw dest_coords: {dest_coords}")
                                continue
        
        # Update global simulation time
        global simulation_time
        simulation_time = float(env.sim.env.now)
        
        # Update training data
        global training_data
        reward = float(self.locals.get('rewards', [0])[-1])
        
        # Track episode reward
        self.current_episode_reward += reward
        
        # Check if episode is done
        if self.locals.get('done', False):
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.step_count)
            self.episode_count += 1
            self.current_episode_reward = 0
        
        training_data['steps'].append(int(self.num_timesteps))
        training_data['rewards'].append(reward)
        training_data['episode_lengths'].append(int(self.locals.get('episode_lengths', [0])[-1]))
        training_data['episode_rewards'].append(self.current_episode_reward)
        
        # Debug logging for episode information
        if self.step_count % self.log_interval == 0:
            print(f"\nEpisode Info:")
            print(f"  Current Reward: {reward}")
            print(f"  Episode Reward: {self.current_episode_reward}")
            print(f"  Episode Count: {self.episode_count}")
            print(f"  Step Count: {self.step_count}")
            print("=== End Debug Info ===\n")
        
        # Emit updates via WebSocket
        socketio.emit('training_update', {
            'steps': int(self.num_timesteps),
            'rewards': reward,
            'episode_reward': self.current_episode_reward,
            'episode_count': self.episode_count,
            'simulation_time': simulation_time,
            'vehicle_count': len(vehicle_states),
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        
        socketio.emit('vehicle_update', {
            'vehicles': list(vehicle_states.values()),
            'requests': request_data
        })
        
        self.step_count += 1
        return True

def main():
    # Start the visualization server
    app = start_visualization_server(port=5003)
    print("Visualization server started at http://localhost:5003")

    # Create simulation parameters
    sim_kwargs = {
        'trip_date': "2019-09-17",
        'simulation_start_time': 7*3600,  # 7 AM
        'simulation_end_time': 10*3600,   # 10 AM
        'accumulation_time': 120,         # 2 minutes decision epoch
        'num_vehicles': 40,
        'vehicle_capacity': 10,
        'randomize_vehicle_position': True
    }

    # Create the environment
    env = VehicleRoutingEnv(sim_kwargs)
    env = DummyVecEnv([lambda: env])

    # Create the model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device='cpu'  # Force CPU usage
    )

    # Train the model
    callback = VisualizationCallback()
    model.learn(
        total_timesteps=100000,
        callback=callback
    )

    # Save the model
    model.save("vehicle_routing_model")

if __name__ == "__main__":
    main()
