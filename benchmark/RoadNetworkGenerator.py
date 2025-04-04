import osmnx as ox
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend
import matplotlib.pyplot as plt

def clean_adjacency_list(adj_list, keys_to_keep):
    """
    Cleans an adjacency list by dropping unnecessary features.

    Parameters:
    - adj_list: dict, adjacency list from nx.to_dict_of_dicts(G)
    - keys_to_keep: list, keys to retain in edge attributes

    Returns:
    - cleaned_adj_list: dict, adjacency list with only specified keys
    """
    cleaned_adj_list = {}
    for node, neighbors in adj_list.items():
        cleaned_adj_list[node] = {}
        for neighbor, edges in neighbors.items():
            cleaned_adj_list[node][neighbor] = {}
            for edge_key, attributes in edges.items():  # Account for multi-edges
                cleaned_attributes = {key: attributes[key] for key in keys_to_keep if key in attributes}
                if cleaned_attributes:  # Only add if attributes are retained
                    cleaned_adj_list[node][neighbor][edge_key] = cleaned_attributes
    return cleaned_adj_list

def extract_maxspeed(maxspeed_list):
    """
    Extract and return the maximum speed from a list of speed strings.
    Parameters:
    - maxspeed_list: list of speed limit strings (e.g., ['40 mph', '50 mph'])

    Returns:
    - max_speed: int, the maximum speed limit in the list (or a default value if unavailable).
    """
    speeds = []
    for speed in maxspeed_list:
        try:
            speeds.append(int(speed.split()[0]))  # Extract numeric part
        except (ValueError, IndexError):
            continue  # Skip invalid entries
    return max(speeds) if speeds else 50  # Default to 50 mph if no valid speed

def generate_sliced_static_network(G, north, south, east, west):
    # calculate travel time - assume 60% average speed
    for u, v, data in G.edges(data=True):
        if "maxspeed" in data:
            if isinstance(data["maxspeed"], list):  # Multiple speeds
                max_speed = extract_maxspeed(data["maxspeed"])
            else:  # Single speed
                max_speed = int(data["maxspeed"].split()[0]) if data["maxspeed"].isdigit() else 50  # Default to 50 mph
        else:
            max_speed = 25  # Default speed if maxspeed is missing

        # Convert max_speed to meters per second
        speed_mps = (max_speed * 0.6 * 1000) / 3600
        if "length" in data:
            data["travel_time"] = data["length"] / speed_mps  # Time in seconds

    # Define the bounding box: north, south, east, west
    north=north
    south=south
    east=east
    west=west  #37.8, 37.77200, -122.38700, -122.42500

    # Get nodes within the bounding box
    nodes_within_bbox = [n for n, d in G.nodes(data=True)
                         if south <= d['y'] <= north and west <= d['x'] <= east]

    # Create a subgraph with these nodes
    G_sliced = G.subgraph(nodes_within_bbox).copy()

    # Convert the osmnx graph to an adjacency list
    adj_list = nx.to_dict_of_dicts(G_sliced)

    # Define keys to keep
    keys_to_keep = ['length', 'geometry', 'lanes', 'oneway', 'reversed', 'maxspeed']

    # Clean the adjacency list
    cleaned_adj_list = clean_adjacency_list(adj_list, keys_to_keep)

    return G_sliced, cleaned_adj_list

# load graph file
# G = ox.load_graphml(filepath="san_francisco.graphml")

# Simplify network
# G = ox.simplify_graph(G)

# sample node coordinate
# G.nodes[32927563]['x']  # Longitude
# G.nodes[32927563]['y']  # Latitude




# Plot the road network
# fig, ax = ox.plot_graph(G_sliced, show=False, close=False)
# fig.savefig("graph_plot_headless.png", dpi=300, bbox_inches="tight")

# Save as GraphML
# ox.save_graphml(G, filepath="san_francisco.graphml")

# Save as Shapefile
# ox.save_graph_shapefile(G, filepath="san_francisco_shapefile/")


# Define positions (q) for vehicles and requests
# vehicle_position = (-122.388, 37.775)  # Example vehicle location (lon, lat)
# request_origin = (-122.4064, 37.7852)    # Example request origin
# request_destination = (-122.3894, 37.7649)  # Example request destination
#
# # Map positions to closest graph nodes
# vehicle_node = ox.nearest_nodes(G, X=vehicle_position[0], Y=vehicle_position[1])
# origin_node = ox.nearest_nodes(G, X=request_origin[0], Y=request_origin[1])
# destination_node = ox.nearest_nodes(G, X=request_destination[0], Y=request_destination[1])
#
# # Calculate travel time between nodes
# travel_time_vehicle_to_origin = nx.shortest_path_length(G, source=vehicle_node, target=origin_node, weight="travel_time")
# travel_time_origin_to_destination = nx.shortest_path_length(G, source=origin_node, target=destination_node, weight="travel_time")
#
# print(f"Travel time (vehicle to origin): {travel_time_vehicle_to_origin / 60:.2f} minutes")
# print(f"Travel time (origin to destination): {travel_time_origin_to_destination / 60:.2f} minutes")

