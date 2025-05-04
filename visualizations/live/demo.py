import os
import random
import time
import threading
import json
import numpy as np
from datetime import datetime
from visualization_server import start_visualization_server, socketio, app, training_data, vehicle_states, request_data

# Configuration
NUM_VEHICLES = 10
VEHICLE_CAPACITY = 10
SIMULATION_DURATION = 600  # 10 minutes in seconds
UPDATE_INTERVAL = 1  # Update every second
SAN_FRANCISCO_BOUNDS = {
    'lat': (37.75, 37.8),  # (min_lat, max_lat)
    'lon': (-122.45, -122.40)  # (min_lon, max_lon)
}

def random_position():
    """Generate a random position within San Francisco bounds"""
    lat = random.uniform(SAN_FRANCISCO_BOUNDS['lat'][0], SAN_FRANCISCO_BOUNDS['lat'][1])
    lon = random.uniform(SAN_FRANCISCO_BOUNDS['lon'][0], SAN_FRANCISCO_BOUNDS['lon'][1])
    return (lat, lon)

def random_route(length=3):
    """Generate a random route with given length"""
    return [random_position() for _ in range(length)]

def generate_random_request():
    """Generate a random passenger request"""
    origin = random_position()
    destination = random_position()
    
    # Ensure origin and destination are not too close
    while np.linalg.norm(np.array(origin) - np.array(destination)) < 0.01:
        destination = random_position()
    
    return {
        'id': random.randint(1000, 9999),
        'origin_lat': origin[0],
        'origin_lon': origin[1],
        'destination_lat': destination[0],
        'destination_lon': destination[1],
        'request_time': time.time(),
        'num_passengers': random.randint(1, 3)
    }

def initialize_vehicles():
    """Initialize random vehicles"""
    global vehicle_states
    
    for i in range(NUM_VEHICLES):
        passengers_count = random.randint(0, VEHICLE_CAPACITY // 2)
        passengers = [f"Passenger-{j}" for j in range(passengers_count)]
        
        vehicle_states[i] = {
            'id': i,
            'position': random_position(),
            'passengers': passengers,
            'capacity': VEHICLE_CAPACITY,
            'route': random_route(random.randint(1, 5)),
            'requests': []
        }

def update_simulation(simulation_time):
    """Update simulation state"""
    global vehicle_states, training_data, request_data
    
    # Update training data
    step = len(training_data['steps']) + 1
    reward = random.uniform(-10, 50)  # Random reward
    
    training_data['steps'].append(step)
    training_data['rewards'].append(reward)
    training_data['episode_lengths'].append(step)
    training_data['episode_rewards'].append(reward)
    
    # Update vehicle positions (simple movement along routes)
    for vehicle_id, data in vehicle_states.items():
        # Move vehicle randomly
        if random.random() < 0.3:  # 30% chance to move
            new_position = random_position()
            # Move slightly towards the next point in route if exists
            if data['route'] and len(data['route']) > 0:
                target = data['route'][0]
                current = data['position']
                # Move 10% towards target
                new_position = (
                    current[0] + 0.1 * (target[0] - current[0]),
                    current[1] + 0.1 * (target[1] - current[1])
                )
                # If close enough to target, remove it from route
                if np.linalg.norm(np.array(new_position) - np.array(target)) < 0.001:
                    data['route'].pop(0)
            
            data['position'] = new_position
        
        # Randomly pick up or drop off passengers
        if random.random() < 0.1:  # 10% chance to change passenger count
            if data['passengers']:
                # Drop off a passenger
                data['passengers'].pop()
            elif len(data['passengers']) < data['capacity']:
                # Pick up a passenger
                data['passengers'].append(f"Passenger-{random.randint(1000, 9999)}")
        
        # Randomly generate new requests
        if random.random() < 0.05:  # 5% chance to get a new request
            data['requests'].append(generate_random_request())
    
    # Generate some random requests for the map
    request_data = [generate_random_request() for _ in range(random.randint(1, 5))]
    
    # Emit data updates via WebSocket
    socketio.emit('training_update', {
        'steps': step,
        'rewards': reward,
        'simulation_time': simulation_time,
        'vehicle_count': len(vehicle_states),
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    socketio.emit('vehicle_update', {
        'vehicles': list(vehicle_states.values()),
        'requests': request_data
    })

def simulation_thread():
    """Run a simulated training process"""
    print("Starting simulation thread...")
    initialize_vehicles()
    
    start_time = time.time()
    
    while time.time() - start_time < SIMULATION_DURATION:
        simulation_time = time.time() - start_time
        update_simulation(simulation_time)
        time.sleep(UPDATE_INTERVAL)
    
    print("Simulation completed!")

if __name__ == "__main__":
    # Start visualization server
    print("Starting visualization server...")
    start_visualization_server(port=5000)
    
    # Start simulation in a separate thread
    threading.Thread(target=simulation_thread).start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping simulation...") 