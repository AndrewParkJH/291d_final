import os
import sys
from stable_baselines3 import PPO
import torch
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from simulator.sav_simulator_rl import ShuttleSim

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)  # Insert at the beginning of the path

import logging
import datetime

log_out_path = os.path.join(project_root, 'output','logs')
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_out_path, f"simulation_{timestamp}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def format_time(seconds):
    """Convert seconds to HH:MM:SS format"""
    return str(timedelta(seconds=int(seconds)))

def log_vehicle_state(vehicle, current_time):
    """Log detailed vehicle state information"""
    logger.info(f"\nVehicle {vehicle.vehicle_id} State at {format_time(current_time)}:")
    logger.info(f"  Current Position: {vehicle.current_pos}")
    logger.info(f"  Current Node: {vehicle.current_node}")
    logger.info(f"  Next Node: {vehicle.next_node}")
    logger.info(f"  Current Passengers: {vehicle.current_num_pax}/{vehicle.max_capacity}")
    logger.info(f"  Trip Sequence Length: {len(vehicle.trip_sequence)}")
    logger.info(f"  Active Requests: {len(vehicle.current_requests)}")
    if vehicle.traversal_process:
        logger.info(f"  Traversal Status: {'Active' if vehicle.traversal_process.is_alive else 'Inactive'}")

def log_trip_info(vehicle, request_id, stage, current_time):
    """Log trip information for a specific request"""
    request = vehicle.current_requests.get(request_id)
    if request:
        if stage == 'pickup':
            request['pickup_time'] = current_time
            logger.info(f"\nTrip {request_id} Started:")
            logger.info(f"  Time: {format_time(current_time)}")
            logger.info(f"  Origin: {request['oid']}")
            logger.info(f"  Destination: {request['did']}")
            logger.info(f"  Passengers: {request['num_passengers']}")
            logger.info(f"  Vehicle ID: {vehicle.vehicle_id}")
            logger.info(f"  Vehicle Position: {vehicle.current_pos}")
        elif stage == 'dropoff':
            request['dropoff_time'] = current_time
            trip_duration = current_time - request['pickup_time']
            logger.info(f"\nTrip {request_id} Completed:")
            logger.info(f"  Completion Time: {format_time(current_time)}")
            logger.info(f"  Trip Duration: {format_time(trip_duration)}")
            logger.info(f"  On-time Status: {'On time' if request['remaining_time'] > 0 else 'Late'}")
            if request['remaining_time'] < 0:
                logger.info(f"  Delay: {format_time(abs(request['remaining_time']))}")
            logger.info(f"  Vehicle ID: {vehicle.vehicle_id}")
            logger.info(f"  Final Position: {vehicle.current_pos}")

def log_vehicle_movement(vehicle, from_node, to_node, travel_time):
    """Log vehicle movement between nodes"""
    logger.info(f"\nVehicle {vehicle.vehicle_id} Movement:")
    logger.info(f"  From Node: {from_node}")
    logger.info(f"  To Node: {to_node}")
    logger.info(f"  Travel Time: {format_time(travel_time)}")
    logger.info(f"  Current Time: {format_time(vehicle.env.now)}")

def plot_rl_stats(trip_stats_rl, rl_accum_times, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Plot total/completed/on-time/late trips for RL
    stats_df = pd.DataFrame([{**{k: v for k, v in trip_stats_rl.items() if k != 'vehicle_stats'}, 'method': 'RL'}])
    melted = stats_df.melt(id_vars='method', value_vars=['total_trips', 'completed_trips', 'on_time_trips', 'late_trips'], var_name='stat', value_name='count')
    plt.figure(figsize=(8,6))
    sns.barplot(data=melted, x='stat', y='count', hue='method')
    plt.title('RL Trip Statistics')
    plt.savefig(os.path.join(output_dir, 'rl_trip_stats.png'))
    plt.close()

    # Plot average delay
    avg_delay = [trip_stats_rl['total_delay']/max(1, trip_stats_rl['late_trips'])]
    plt.figure(figsize=(6,4))
    sns.barplot(x=['RL'], y=avg_delay)
    plt.title('RL Average Delay (Late Trips Only)')
    plt.ylabel('Delay (seconds)')
    plt.savefig(os.path.join(output_dir, 'rl_avg_delay.png'))
    plt.close()

    # Accumulated time to serve number of requests
    plt.figure(figsize=(8,6))
    plt.plot(rl_accum_times['served_requests'], rl_accum_times['accum_time'], label='RL')
    plt.xlabel('Number of Requests Served')
    plt.ylabel('Accumulated Time (seconds)')
    plt.title('RL Accumulated Time to Serve Requests')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'rl_accum_time_vs_requests.png'))
    plt.close()

def plot_stats(trip_stats_rl, trip_stats_ilp, rl_accum_times, ilp_accum_times, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Plot total/completed/on-time/late trips for each method
    stats_df = pd.DataFrame([
        {**{k: v for k, v in trip_stats_rl.items() if k != 'vehicle_stats'}, 'method': 'RL'},
        {**{k: v for k, v in trip_stats_ilp.items() if k != 'vehicle_stats'}, 'method': 'ILP'}
    ])
    melted = stats_df.melt(id_vars='method', value_vars=['total_trips', 'completed_trips', 'on_time_trips', 'late_trips'], var_name='stat', value_name='count')
    plt.figure(figsize=(8,6))
    sns.barplot(data=melted, x='stat', y='count', hue='method')
    plt.title('Trip Statistics by Method')
    plt.savefig(os.path.join(output_dir, 'trip_stats_comparison.png'))
    plt.close()

    # Plot average delay
    avg_delay = [trip_stats_rl['total_delay']/max(1, trip_stats_rl['late_trips']), trip_stats_ilp['total_delay']/max(1, trip_stats_ilp['late_trips'])]
    plt.figure(figsize=(6,4))
    sns.barplot(x=['RL', 'ILP'], y=avg_delay)
    plt.title('Average Delay (Late Trips Only)')
    plt.ylabel('Delay (seconds)')
    plt.savefig(os.path.join(output_dir, 'avg_delay_comparison.png'))
    plt.close()

    # Accumulated time to serve number of requests
    plt.figure(figsize=(8,6))
    plt.plot(rl_accum_times['served_requests'], rl_accum_times['accum_time'], label='RL')
    plt.plot(ilp_accum_times['served_requests'], ilp_accum_times['accum_time'], label='ILP')
    plt.xlabel('Number of Requests Served')
    plt.ylabel('Accumulated Time (seconds)')
    plt.title('Accumulated Time to Serve Requests')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accum_time_vs_requests.png'))
    plt.close()

def main():
    # Simulation parameters for RL mode
    sim_kwargs_rl = {
        'run_mode': "rl",
        'trip_date': "2019-09-17",
        'simulation_start_time': 7 * 3600,  # 7 AM
        'simulation_end_time': 10 * 3600,  # 10 AM
        'accumulation_time': 120,  # 2 minutes decision epoch
        'num_vehicles': 1,
        'vehicle_capacity': 10,
        'randomize_vehicle_position': True,
        'euclidean_radius': 800,
        'walking_speed': 1.2,
        'max_walk_time': 600
    }

    # Simulation parameters for ILP mode
    sim_kwargs_ilp = {
        'run_mode': "ilp",
        'trip_date': "2019-09-17",
        'simulation_start_time': 7 * 3600,  # 7 AM
        'simulation_end_time': 10 * 3600,  # 10 AM
        'accumulation_time': 120,  # 2 minutes decision epoch
        'num_vehicles': 1,
        'vehicle_capacity': 10,
        'randomize_vehicle_position': True,
        'euclidean_radius': 800,
        'walking_speed': 1.2,
        'max_walk_time': 600
    }

    action_encoder = sim_kwargs_rl['vehicle_capacity'] * 2
    decision_epoch = sim_kwargs_rl['accumulation_time']

    # Initialize simulators
    simulator_rl = ShuttleSim(**sim_kwargs_rl)
    simpy_env_rl = simulator_rl.reset_simulator()

    simulator_ilp = ShuttleSim(**sim_kwargs_ilp)
    simpy_env_ilp = simulator_ilp.reset_simulator()

    # Force CPU usage for PyTorch
    device = torch.device("cpu")
    torch.set_num_threads(20)  # Adjust based on your CPU cores

    # Load model from checkpoint
    checkpoint_path = "/home/rishi/Berkeley/291d_final/RL/training/logs/PPO20/checkpoint_log"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
    
    # Find the latest model file in the checkpoint directory
    model_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.zip')]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {checkpoint_path}")
    
    latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)))
    model_path = os.path.join(checkpoint_path, latest_model)
    logger.info(f"Loading model from: {model_path}")
    
    rl_model = PPO.load(model_path, device=device)

    # Initialize trip statistics
    trip_stats_rl = {
        'total_trips': 0,
        'completed_trips': 0,
        'on_time_trips': 0,
        'late_trips': 0,
        'total_delay': 0,
        'vehicle_stats': {}
    }
    trip_stats_ilp = {
        'total_trips': 0,
        'completed_trips': 0,
        'on_time_trips': 0,
        'late_trips': 0,
        'total_delay': 0,
        'vehicle_stats': {}
    }

    logger.info("Starting simulation...")
    logger.info(f"Simulation time: {format_time(sim_kwargs_rl['simulation_start_time'])} - {format_time(sim_kwargs_rl['simulation_end_time'])}")

    rl_accum_times = {'served_requests': [], 'accum_time': []}
    ilp_accum_times = {'served_requests': [], 'accum_time': []}

    # Run RL simulation
    served_rl = 0
    while simulator_rl.env.now < sim_kwargs_rl['simulation_end_time']:
        # Get observations for all vehicles
        obs, info = simulator_rl.get_observation(agent_object='vehicle')
        
        # Log current state
        logger.info(f"\nSimulation time: {format_time(simulator_rl.env.now)}")
        logger.info(f"Number of pending requests: {len(simulator_rl.current_request)}")
        
        # Get actions from RL model for each vehicle
        actions = []
        for i, vehicle_obs in enumerate(obs):
            # Reshape observation to match expected shape (1, observation_dim)
            reshaped_obs = np.array(vehicle_obs).reshape(1, -1)
            action, _ = rl_model.predict(reshaped_obs, deterministic=True)
            # Decode the action into pickup and dropoff indices
            decoded_action = decode_action(action[0], action_encoder)  # Get first action since we reshaped
            actions.append(decoded_action)
            logger.debug(f"Vehicle {i} action: {action} -> decoded: {decoded_action}")
        
        # Apply actions to assign requests
        invalid_flags = simulator_rl.apply_actions(agent_object='vehicle', actions=actions)
        
        # Log vehicle states and trip information
        for vehicle in simulator_rl.network.vehicles:
            # Initialize vehicle stats if not exists
            if vehicle.vehicle_id not in trip_stats_rl['vehicle_stats']:
                trip_stats_rl['vehicle_stats'][vehicle.vehicle_id] = {
                    'total_trips': 0,
                    'completed_trips': 0,
                    'on_time_trips': 0,
                    'late_trips': 0,
                    'total_delay': 0
                }
            
            # Log vehicle state
            log_vehicle_state(vehicle, simulator_rl.env.now)
            
            # Track vehicle movement
            if vehicle.current_node != vehicle.next_node:
                log_vehicle_movement(vehicle, vehicle.current_node, vehicle.next_node, 
                                   vehicle.current_segment_times[0] if vehicle.current_segment_times else 0)
            
            # Log trip events
            for request_id, request in vehicle.current_requests.items():
                # Ensure request has all required fields
                if 'pickup_time' not in request:
                    request['pickup_time'] = -1
                if 'dropoff_time' not in request:
                    request['dropoff_time'] = -1
                if 'remaining_time' not in request:
                    request['remaining_time'] = request.get('deadline', float('inf')) - simulator_rl.env.now
                
                # Check for pickup
                if request['pickup_time'] == -1 and vehicle.current_node == request.get('oid', request.get('pu_osmid')):
                    log_trip_info(vehicle, request_id, 'pickup', simulator_rl.env.now)
                    trip_stats_rl['total_trips'] += 1
                    trip_stats_rl['vehicle_stats'][vehicle.vehicle_id]['total_trips'] += 1
                    served_rl += 1
                    rl_accum_times['served_requests'].append(served_rl)
                    rl_accum_times['accum_time'].append(simulator_rl.env.now)
                
                # Check for dropoff
                elif request['dropoff_time'] == -1 and vehicle.current_node == request.get('did', request.get('do_osmid')):
                    log_trip_info(vehicle, request_id, 'dropoff', simulator_rl.env.now)
                    trip_stats_rl['completed_trips'] += 1
                    trip_stats_rl['vehicle_stats'][vehicle.vehicle_id]['completed_trips'] += 1
                    if request['remaining_time'] > 0:
                        trip_stats_rl['on_time_trips'] += 1
                        trip_stats_rl['vehicle_stats'][vehicle.vehicle_id]['on_time_trips'] += 1
                    else:
                        trip_stats_rl['late_trips'] += 1
                        trip_stats_rl['vehicle_stats'][vehicle.vehicle_id]['late_trips'] += 1
                        delay = abs(request['remaining_time'])
                        trip_stats_rl['total_delay'] += delay
                        trip_stats_rl['vehicle_stats'][vehicle.vehicle_id]['total_delay'] += delay

        # Update simulation state
        simulator_rl.update_state(agent_object='vehicle', simulation_run_time=decision_epoch)

    # Plot RL stats before running ILP
    plot_rl_stats(trip_stats_rl, rl_accum_times, os.path.join(project_root, 'output', 'plots'))

    # Run ILP simulation
    served_ilp = 0
    while simulator_ilp.env.now < sim_kwargs_ilp['simulation_end_time']:
        simulator_ilp.update_state(agent_object='vehicle', simulation_run_time=decision_epoch)
        simulator_ilp.solve_vrp_ilp()
        # Track number of requests served and time
        # Collect stats for ILP vehicles
        for vehicle in simulator_ilp.network.vehicles:
            if vehicle.vehicle_id not in trip_stats_ilp['vehicle_stats']:
                trip_stats_ilp['vehicle_stats'][vehicle.vehicle_id] = {
                    'total_trips': 0,
                    'completed_trips': 0,
                    'on_time_trips': 0,
                    'late_trips': 0,
                    'total_delay': 0
                }
            for request_id, request in vehicle.current_requests.items():
                if 'pickup_time' not in request:
                    request['pickup_time'] = -1
                if 'dropoff_time' not in request:
                    request['dropoff_time'] = -1
                if 'remaining_time' not in request:
                    request['remaining_time'] = request.get('deadline', float('inf')) - simulator_ilp.env.now
                if request['pickup_time'] == -1 and vehicle.current_node == request.get('oid', request.get('pu_osmid')):
                    trip_stats_ilp['total_trips'] += 1
                    trip_stats_ilp['vehicle_stats'][vehicle.vehicle_id]['total_trips'] += 1
                    served_ilp += 1
                    ilp_accum_times['served_requests'].append(served_ilp)
                    ilp_accum_times['accum_time'].append(simulator_ilp.env.now)
                elif request['dropoff_time'] == -1 and vehicle.current_node == request.get('did', request.get('do_osmid')):
                    trip_stats_ilp['completed_trips'] += 1
                    trip_stats_ilp['vehicle_stats'][vehicle.vehicle_id]['completed_trips'] += 1
                    if request['remaining_time'] > 0:
                        trip_stats_ilp['on_time_trips'] += 1
                        trip_stats_ilp['vehicle_stats'][vehicle.vehicle_id]['on_time_trips'] += 1
                    else:
                        trip_stats_ilp['late_trips'] += 1
                        trip_stats_ilp['vehicle_stats'][vehicle.vehicle_id]['late_trips'] += 1
                        delay = abs(request['remaining_time'])
                        trip_stats_ilp['total_delay'] += delay
                        trip_stats_ilp['vehicle_stats'][vehicle.vehicle_id]['total_delay'] += delay

    # Log final statistics
    logger.info("\nSimulation completed!")
    logger.info("Final Statistics:")
    logger.info(f"Total trips: {trip_stats_rl['total_trips']}")
    logger.info(f"Completed trips: {trip_stats_rl['completed_trips']}")
    logger.info(f"On-time trips: {trip_stats_rl['on_time_trips']}")
    logger.info(f"Late trips: {trip_stats_rl['late_trips']}")
    logger.info(f"Average delay: {format_time(trip_stats_rl['total_delay'] / max(1, trip_stats_rl['late_trips']))}")
    
    # Log per-vehicle statistics
    logger.info("\nPer-Vehicle Statistics:")
    for vehicle_id, stats in trip_stats_rl['vehicle_stats'].items():
        logger.info(f"\nVehicle {vehicle_id}:")
        logger.info(f"  Total trips: {stats['total_trips']}")
        logger.info(f"  Completed trips: {stats['completed_trips']}")
        logger.info(f"  On-time trips: {stats['on_time_trips']}")
        logger.info(f"  Late trips: {stats['late_trips']}")
        if stats['late_trips'] > 0:
            logger.info(f"  Average delay: {format_time(stats['total_delay'] / stats['late_trips'])}")

    # Plot and compare
    plot_stats(trip_stats_rl, trip_stats_ilp, rl_accum_times, ilp_accum_times, os.path.join(project_root, 'output', 'plots'))

def decode_action(action, action_encoder):
    """Decode the action from the model into pickup and dropoff indices"""
    i = action // action_encoder
    j = action % action_encoder
    return i, j

if __name__ == "__main__":
    main()