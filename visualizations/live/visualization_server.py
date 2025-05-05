import os
import json
import numpy as np
import pandas as pd
import threading
import time
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory
from flask_socketio import SocketIO
from stable_baselines3.common.callbacks import BaseCallback
import folium
from folium.plugins import HeatMap
import networkx as nx
import osmnx as ox

# Configuration
LOG_INTERVAL = 1  # Log every N training steps
GRAPH_FILE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data/road_network/sf_road_network.graphml")

# Global variables for storing visualization data
training_data = {
    'steps': [],
    'rewards': [],
    'episode_lengths': [],
    'episode_rewards': []
}
vehicle_states = {}
request_data = []
simulation_time = 0

# Create Flask application
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load road network
try:
    graph = ox.load_graphml(GRAPH_FILE_DIR)
    print(f"Road network loaded with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
except Exception as e:
    print(f"Error loading road network: {e}")
    graph = None

class VisualizationCallback(BaseCallback):
    """
    Custom callback for capturing training metrics and vehicle states
    """
    def __init__(self, log_interval=LOG_INTERVAL, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval
        
    def _on_step(self) -> bool:
        """
        This method will be called every step of the model's training
        """
        global training_data, vehicle_states, simulation_time, request_data
        
        if self.n_calls % self.log_interval == 0:
            # Get current metrics
            rewards = self.model.ep_info_buffer[-1]["r"] if len(self.model.ep_info_buffer) > 0 else 0
            ep_len = self.model.ep_info_buffer[-1]["l"] if len(self.model.ep_info_buffer) > 0 else 0
            
            # Store metrics
            training_data['steps'].append(self.num_timesteps)
            training_data['rewards'].append(rewards)
            training_data['episode_lengths'].append(ep_len)
            training_data['episode_rewards'].append(rewards)
            
            # Get vehicle states from the environment
            if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs'):
                try:
                    env = self.model.env.envs[0]
                    if hasattr(env, 'sim'):
                        sim = env.sim
                        simulation_time = sim.env.now if hasattr(sim, 'env') else 0
                        
                        # Clear previous vehicle states
                        vehicle_states.clear()
                        
                        # Update vehicle states
                        if hasattr(sim, 'network') and hasattr(sim.network, 'vehicles'):
                            for i, vehicle in enumerate(sim.network.vehicles):
                                try:
                                    # Try to get current position
                                    position = None
                                    if hasattr(vehicle, 'current_position'):
                                        position = vehicle.current_position
                                    elif hasattr(vehicle, 'position'):
                                        position = vehicle.position
                                    elif hasattr(vehicle, 'pos'):
                                        position = vehicle.pos
                                    elif hasattr(vehicle, 'current_pos'):
                                        position = vehicle.current_pos
                                    else:
                                        # Log this issue for debugging
                                        print(f"Warning: Could not find position attribute for vehicle {i}")
                                        
                                    # If position is actually lon,lat (like in vehicle_rl.py), swap to lat,lon for mapping
                                    if position and isinstance(position, tuple) and len(position) == 2:
                                        # Check if first value looks like longitude (larger negative number)
                                        if position[0] < 0 and position[0] < -100:
                                            # This is (lon, lat) format, swap to (lat, lon) for mapping
                                            position = (position[1], position[0])
                                        
                                        # Print first vehicle position for debugging
                                        if i == 0 and self.n_calls % (self.log_interval * 10) == 0:
                                            print(f"Vehicle 0 position after formatting: {position}")
                                    
                                    # Try to get passenger info
                                    passengers = []
                                    if hasattr(vehicle, 'current_passengers'):
                                        passengers = vehicle.current_passengers
                                    elif hasattr(vehicle, 'passengers'):
                                        passengers = vehicle.passengers
                                    
                                    # Try to get capacity
                                    capacity = 0
                                    if hasattr(vehicle, 'capacity'):
                                        capacity = vehicle.capacity
                                    
                                    # Try to get route
                                    route = []
                                    if hasattr(vehicle, 'route'):
                                        route = vehicle.route
                                    
                                    # Try to get requests
                                    vehicle_requests = []
                                    if hasattr(vehicle, 'requests'):
                                        vehicle_requests = vehicle.requests
                                        
                                    # Debug information
                                    if i == 0 and self.n_calls % (self.log_interval * 10) == 0:
                                        print(f"Vehicle 0 data - Position: {position}, Capacity: {capacity}")
                                    
                                    # Create vehicle data dictionary
                                    vehicle_data = {
                                        'id': i,
                                        'position': position,
                                        'passengers': passengers if passengers is not None else [],
                                        'capacity': capacity,
                                        'route': route if route is not None else [],
                                        'requests': vehicle_requests if vehicle_requests is not None else []
                                    }
                                    vehicle_states[i] = vehicle_data
                                except Exception as e:
                                    print(f"Error capturing vehicle {i} data: {e}")
                        
                        # Update request data
                        current_requests = []
                        if hasattr(sim, 'current_request'):
                            current_requests = sim.current_request
                            if current_requests:
                                # Process the request data to have the right format for visualization
                                processed_requests = []
                                for req in current_requests:
                                    try:
                                        # Check if it's already a dictionary
                                        if isinstance(req, dict):
                                            processed_req = req
                                        # Or if it has attributes we can access
                                        elif hasattr(req, 'to_dict'):
                                            processed_req = req.to_dict()
                                        # Extract information if possible
                                        elif hasattr(req, 'origin') and hasattr(req, 'destination'):
                                            processed_req = {
                                                'id': getattr(req, 'id', 'unknown'),
                                                'origin_lat': req.origin[0] if isinstance(req.origin, tuple) else None,
                                                'origin_lon': req.origin[1] if isinstance(req.origin, tuple) else None,
                                                'destination_lat': req.destination[0] if isinstance(req.destination, tuple) else None,
                                                'destination_lon': req.destination[1] if isinstance(req.destination, tuple) else None
                                            }
                                        else:
                                            # Skip if we can't extract information
                                            continue
                                            
                                        processed_requests.append(processed_req)
                                    except Exception as e:
                                        print(f"Error processing request: {e}")
                                
                                request_data = processed_requests
                except Exception as e:
                    print(f"Error in visualization callback: {e}")
            
            # Emit updates via WebSocket
            emit_updates()
            
        return True

def emit_updates():
    """Emit updates to all connected clients"""
    socketio.emit('training_update', {
        'steps': training_data['steps'][-1] if training_data['steps'] else 0,
        'rewards': training_data['rewards'][-1] if training_data['rewards'] else 0,
        'episode_reward': training_data['episode_rewards'][-1] if training_data['episode_rewards'] else 0,
        'episode_count': len(training_data['episode_rewards']),
        'simulation_time': simulation_time,
        'vehicle_count': len(vehicle_states),
        'timestamp': datetime.now().strftime('%H:%M:%S')
    })
    
    # Emit vehicle updates
    socketio.emit('vehicle_update', {
        'vehicles': list(vehicle_states.values()),
        'requests': request_data
    })

def generate_map():
    """Generate folium map with current vehicle positions"""
    # Create map centered on San Francisco, even if graph is None
    m = folium.Map(location=[37.7749, -122.4194], zoom_start=12, tiles='cartodbpositron')
    
    # Add vehicles to map
    for vehicle_id, data in vehicle_states.items():
        pos = data.get('position')
        if isinstance(pos, tuple) and len(pos) == 2:
            # Add marker for vehicle
            folium.CircleMarker(
                location=[pos[1], pos[0]],  # Swap lat/lon order
                radius=5,
                color='blue',
                fill=True,
                fill_opacity=0.7,
                popup=f"Vehicle {vehicle_id}: {data.get('passengers', []).__len__()}/{data.get('capacity')} passengers"
            ).add_to(m)
    
    # Add requests to map
    for req in request_data:
        if isinstance(req, dict) and 'origin_lat' in req and 'origin_lon' in req:
            # Add marker for request origin
            folium.CircleMarker(
                location=[req['origin_lat'], req['origin_lon']],  # Already in correct order
                radius=3,
                color='red',
                fill=True,
                fill_opacity=0.7,
                popup=f"Request: {req.get('id', 'Unknown')}"
            ).add_to(m)
            
            # Add marker for request destination
            if 'destination_lat' in req and 'destination_lon' in req:
                folium.CircleMarker(
                    location=[req['destination_lat'], req['destination_lon']],  # Already in correct order
                    radius=3,
                    color='green',
                    fill=True,
                    fill_opacity=0.7
                ).add_to(m)
                
                # Add line between origin and destination
                folium.PolyLine(
                    locations=[[req['origin_lat'], req['origin_lon']], 
                              [req['destination_lat'], req['destination_lon']]],  # Already in correct order
                    color='gray',
                    weight=2,
                    opacity=0.5
                ).add_to(m)
    
    return m

@app.route('/')
def index():
    """Render main visualization dashboard"""
    return render_template('index.html')

@app.route('/api/training_data')
def get_training_data():
    """Return all training data for charts"""
    return jsonify(training_data)

@app.route('/api/vehicle_states')
def get_vehicle_states():
    """Return current vehicle states"""
    return jsonify({
        'vehicles': list(vehicle_states.values()),
        'simulation_time': simulation_time,
        'requests': request_data
    })

@app.route('/debug')
def debug_info():
    """Provide debugging information"""
    debug_data = {
        'training_data': {
            'steps_count': len(training_data['steps']),
            'latest_step': training_data['steps'][-1] if training_data['steps'] else None,
            'latest_reward': training_data['rewards'][-1] if training_data['rewards'] else None
        },
        'vehicle_states': {
            'count': len(vehicle_states),
            'sample': next(iter(vehicle_states.values())) if vehicle_states else None
        },
        'request_data': {
            'count': len(request_data),
            'sample': request_data[0] if request_data else None
        },
        'simulation_time': simulation_time
    }
    return jsonify(debug_data)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial data
    emit_updates()

def start_visualization_server(port=5000):
    """Start the visualization server in a separate thread"""
    threading.Thread(target=lambda: socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)).start()
    print(f"Visualization server started at http://localhost:{port}")
    return app

# If run directly, start the server
if __name__ == '__main__':
    start_visualization_server() 