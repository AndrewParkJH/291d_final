import os
import sys
from stable_baselines3 import PPO
import torch
import pandas as pd
from datetime import datetime, timedelta

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
    torch.set_num_threads(4)  # Adjust based on your CPU cores

    # Load model from checkpoint
    checkpoint_path = "/home/rishi/Berkeley/291d_final/RL/training/logs/PPO15/checkpoint_log"
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
    trip_stats = {
        'total_trips': 0,
        'completed_trips': 0,
        'on_time_trips': 0,
        'late_trips': 0,
        'total_delay': 0,
        'vehicle_stats': {}  # Per-vehicle statistics
    }

    logger.info("Starting simulation...")
    logger.info(f"Simulation time: {format_time(sim_kwargs_rl['simulation_start_time'])} - {format_time(sim_kwargs_rl['simulation_end_time'])}")

    # Run RL simulation
    while simulator_rl.env.now < sim_kwargs_rl['simulation_end_time']:
        obs, info = simulator_rl.get_observation(agent_object='vehicle')
        action, _ = rl_model.predict(obs, deterministic=True)  # Using deterministic=True for evaluation
        
        # Decode action for each vehicle
        decoded_actions = []
        for vehicle_action in action:
            decoded_action = decode_action(vehicle_action, action_encoder)
            decoded_actions.append(decoded_action)
            
        simulator_rl.apply_actions(agent_object='vehicle', actions=decoded_actions)
        
        # Log vehicle states and trip information
        for vehicle in simulator_rl.network.vehicles:
            # Initialize vehicle stats if not exists
            if vehicle.vehicle_id not in trip_stats['vehicle_stats']:
                trip_stats['vehicle_stats'][vehicle.vehicle_id] = {
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
                if request['pickup_time'] == -1 and vehicle.current_node == request['oid']:
                    log_trip_info(vehicle, request_id, 'pickup', simulator_rl.env.now)
                    trip_stats['total_trips'] += 1
                    trip_stats['vehicle_stats'][vehicle.vehicle_id]['total_trips'] += 1
                elif request['dropoff_time'] == -1 and vehicle.current_node == request['did']:
                    log_trip_info(vehicle, request_id, 'dropoff', simulator_rl.env.now)
                    trip_stats['completed_trips'] += 1
                    trip_stats['vehicle_stats'][vehicle.vehicle_id]['completed_trips'] += 1
                    if request['remaining_time'] > 0:
                        trip_stats['on_time_trips'] += 1
                        trip_stats['vehicle_stats'][vehicle.vehicle_id]['on_time_trips'] += 1
                    else:
                        trip_stats['late_trips'] += 1
                        trip_stats['vehicle_stats'][vehicle.vehicle_id]['late_trips'] += 1
                        delay = abs(request['remaining_time'])
                        trip_stats['total_delay'] += delay
                        trip_stats['vehicle_stats'][vehicle.vehicle_id]['total_delay'] += delay

        simulator_rl.update_state(agent_object='vehicle', simulation_run_time=decision_epoch)

    # Run ILP simulation
    while simulator_ilp.env.now < sim_kwargs_ilp['simulation_end_time']:
        simulator_ilp.update_state(agent_object='vehicle', simulation_run_time=decision_epoch)
        simulator_ilp.solve_vrp_ilp()

    # Log final statistics
    logger.info("\nSimulation completed!")
    logger.info("Final Statistics:")
    logger.info(f"Total trips: {trip_stats['total_trips']}")
    logger.info(f"Completed trips: {trip_stats['completed_trips']}")
    logger.info(f"On-time trips: {trip_stats['on_time_trips']}")
    logger.info(f"Late trips: {trip_stats['late_trips']}")
    logger.info(f"Average delay: {format_time(trip_stats['total_delay'] / max(1, trip_stats['late_trips']))}")
    
    # Log per-vehicle statistics
    logger.info("\nPer-Vehicle Statistics:")
    for vehicle_id, stats in trip_stats['vehicle_stats'].items():
        logger.info(f"\nVehicle {vehicle_id}:")
        logger.info(f"  Total trips: {stats['total_trips']}")
        logger.info(f"  Completed trips: {stats['completed_trips']}")
        logger.info(f"  On-time trips: {stats['on_time_trips']}")
        logger.info(f"  Late trips: {stats['late_trips']}")
        if stats['late_trips'] > 0:
            logger.info(f"  Average delay: {format_time(stats['total_delay'] / stats['late_trips'])}")

def decode_action(action, action_encoder):
    """Decode the action from the model into pickup and dropoff indices"""
    i = action // action_encoder
    j = action % action_encoder
    return i, j

if __name__ == "__main__":
    main()