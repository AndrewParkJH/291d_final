import networkx as nx
from benchmark.RVGraphGenerator import travel

def generate_rtv_graph(rv_graph, vehicles, requests, graph, max_capacity=2, max_delay=600, debug=False):
    """
    Generate the Request-Trip-Vehicle (RTV) graph.

    Parameters:
    - rv_graph: networkx.Graph, the RV graph generated in the previous step.
    - vehicles: list of dicts, each representing a vehicle.
    - requests: list of dicts, each representing a request.
    - graph: networkx.Graph, the road network.
    - max_capacity: int, maximum passenger capacity per vehicle.
    - max_delay: int, maximum allowable delay time in seconds.

    Returns:
    - rtv_graph: networkx.Graph, the generated RTV graph.
    """
    import itertools

    rtv_graph = nx.Graph()

    # Step 1: Add nodes for requests, trips, and vehicles
    for request in requests:
        rtv_graph.add_node(request["id"], type="request", **request)

    for vehicle in vehicles:
        rtv_graph.add_node(vehicle["id"], type="vehicle", **vehicle)

    # Step 2: Generate trips of increasing sizes
    for vehicle in vehicles:
        vehicle_id = vehicle["id"]
        vehicle_position = vehicle["q_v"]
        vehicle_time = vehicle["t_v"]

        if debug:
            print(f'checking vid: {vehicle_id}:')

        # Start with trips of size 1 (direct VR edges)
        trips = [{request_id} for _, request_id, edge_data in rv_graph.edges(vehicle_id, data=True)
                 if rv_graph.nodes[request_id]["type"] == "request"]

        trip_size = 1
        all_valid_trips = []

        if debug:
            print(f'...checking trips {trips} for {vehicle_id}')

        while trips and trip_size <= max_capacity:
            if debug:
                print(f'\t vid_{vehicle_id}-trip_size: {trip_size}')
            new_trips = []

            for trip in trips:

                trip_requests = [next(req for req in requests if req["id"] == r_id) for r_id in trip]

                if debug:
                    print(f'\t \t trip: {trip} \t trip_requests: {trip_requests} -- calling travel({vehicle_id}, trip_request)')
                # Validate the trip for this vehicle using `travel`
                valid_trip, trip_cost = travel(vehicle, trip_requests, graph, max_capacity, max_delay)#, debug)

                if valid_trip:
                    # Add trip as a node in the RTV graph
                    trip_node_id = f"T_{vehicle_id}_{len(all_valid_trips)}"  # Unique ID for the trip
                    trip_data = {
                        "requests": trip,
                        "vehicle": vehicle_id,
                        "cost": trip_cost,
                        "stops": valid_trip
                    }
                    rtv_graph.add_node(trip_node_id, type="trip", **trip_data)

                    # Connect requests to the trip
                    for r_id in trip:
                        rtv_graph.add_edge(r_id, trip_node_id, edge_type="rt", travel_time=trip_cost)

                    # Connect the trip to the vehicle
                    rtv_graph.add_edge(trip_node_id, vehicle_id, edge_type="tv", travel_time=trip_cost)

                    all_valid_trips.append(trip)

            # Generate larger trips by combining smaller trips
            trip_size += 1
            new_trips = [trip1.union(trip2) for trip1, trip2 in itertools.combinations(all_valid_trips, 2)
                         if len(trip1.union(trip2)) == trip_size]

            trips = new_trips  # Update the trips for the next iteration

    return rtv_graph

def greedy_assignment(rtv_graph, vehicles, debug=False):
    """
    Perform greedy assignment of trips to vehicles based on Algorithm 2.

    Parameters:
    - rtv_graph: networkx.Graph, the RTV graph containing trips, vehicles, and requests.
    - vehicles: list of dicts, vehicle attributes.
    - requests: list of dicts, request attributes.

    Returns:
    - assignment: dict, mapping of vehicle IDs to trip IDs.
    """
    assignment = {}
    # only include vehicles with non-zero degrees
    unassigned_vehicles = {vehicle["id"] for vehicle in vehicles if rtv_graph.degree(vehicle["id"]) > 0}
    assigned_requests = set()

    # Iterate until no vehicles or trips can be assigned
    while unassigned_vehicles:
        best_vehicle = None
        best_trip = None
        min_cost = float("inf")

        # Iterate over all vehicles and trips in the RTV graph
        for vehicle_id in unassigned_vehicles:
            neighbors = list(rtv_graph.neighbors(vehicle_id))

            for trip_id in neighbors:
                if rtv_graph.nodes[trip_id]["type"] == "trip":

                    # Check if trip involves already assigned requests
                    trip_requests = rtv_graph.nodes[trip_id]["requests"]
                    number_of_trips = len(trip_requests)

                    if trip_requests & assigned_requests:
                        continue  # Skip trips with already assigned requests

                    # Calculate the cost of assigning this trip to the vehicle
                    edge_data = rtv_graph.get_edge_data(vehicle_id, trip_id, {})
                    cost = edge_data.get("travel_time", float("inf"))
                    relative_cost = cost/number_of_trips # greedily look for a vehicle that can support multiple trips

                    if relative_cost < min_cost:
                        min_cost = relative_cost
                        best_vehicle = vehicle_id
                        best_trip = trip_id

                        if debug and vehicle_id == 'v3':
                                print(
                                    f'{vehicle_id} - {trip_id} trip assigned \t\t'
                                    f'length: {len(rtv_graph.nodes[trip_id]["requests"])} '
                                    f'trips: {rtv_graph.nodes[trip_id]["requests"]} '
                                    f'cost: {rtv_graph.nodes[trip_id]["cost"]} '
                                    f'relative cost: {relative_cost}')

        # Assign the best trip to the best vehicle
        if best_vehicle and best_trip:
            assignment[best_vehicle] = best_trip
            unassigned_vehicles.remove(best_vehicle)

            # Mark requests in the assigned trip as served
            for request_id in rtv_graph.neighbors(best_trip):
                if rtv_graph.nodes[request_id]["type"] == "request":
                    assigned_requests.add(request_id)
        else:
            break  # No more feasible assignments can be made

    return assignment

def visualize_assignment(graph_road_map, vehicles, requests, assignment, rtv_graph, filepath="greedy_assignment_visualization.png"):
    """
    Visualize the assignment of trips to vehicles over the road network.

    Parameters:
    - G_edt: networkx.Graph, the road network.
    - vehicles: list of dicts, vehicle attributes.
    - requests: list of dicts, request attributes.
    - assignment: dict, mapping of vehicle IDs to trip IDs.
    - rtv_graph: networkx.Graph, the RTV graph used for assignment.
    - filepath: str, file path to save the visualization.

    Returns:
    - None (saves the visualization to the specified file path).
    """
    import matplotlib.pyplot as plt
    import osmnx as ox

    # Base road network visualization
    fig, ax = ox.plot_graph(graph_road_map, show=False, close=False, node_size=5, edge_color="gray", bgcolor="white")

    # Get positions for road network nodes
    pos = {node: (data['x'], data['y']) for node, data in graph_road_map.nodes(data=True)}

    # Plot vehicles as blue circles
    for vehicle in vehicles:
        vehicle_pos = pos.get(vehicle["q_v"])
        if vehicle_pos:
            ax.scatter(vehicle_pos[0], vehicle_pos[1], c="blue", s=50, label="Vehicle" if "Vehicle" not in ax.get_legend_handles_labels()[1] else None)
            ax.text(vehicle_pos[0], vehicle_pos[1], f"{vehicle['id']}-p#{vehicle['passengers']}", fontsize=8, color="black", ha="left", va="bottom")

    # Plot requests as red circles
    for request in requests:
        request_pos = pos.get(request["o_r"])
        if request_pos:
            ax.scatter(request_pos[0], request_pos[1], c="red", s=30, label="Request" if "Request" not in ax.get_legend_handles_labels()[1] else None)


    # Visualize assigned routes
    for vehicle_id, trip_id in assignment.items():
        if trip_id not in rtv_graph:
            continue

        # Get the trip stop sequence
        trip_data = rtv_graph.nodes[trip_id]
        trip_stops = trip_data.get("stops", [])  # Ensure the trip has stops
        if not trip_stops:
            continue

        # Get the trip stop sequence
        trip_data = rtv_graph.nodes[trip_id]
        trip_requests = trip_data.get("requests", {})   # Ensure the trip has stops
        if not trip_requests:
            continue

        # Draw the route based on the trip stop order
        current_pos = vehicles[[v["id"] for v in vehicles].index(vehicle_id)]["q_v"]  # Vehicle's current position
        for stop in trip_stops:

            next_pos = stop["node"]  # Stop position
            if current_pos in pos and next_pos in pos:
                current_pos_xy = pos[current_pos]
                next_pos_xy = pos[next_pos]
                ax.plot(
                    [current_pos_xy[0], next_pos_xy[0]],
                    [current_pos_xy[1], next_pos_xy[1]],
                    c="red", linestyle="--", alpha=0.7, linewidth=1,
                )
            current_pos = next_pos

    # Add legend and save the figure
    ax.legend(loc="upper left")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


