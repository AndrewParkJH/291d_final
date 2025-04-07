from itertools import permutations
import networkx as nx
from itertools import combinations
import heapq
import osmnx as ox
import matplotlib.pyplot as plt
from fontTools.merge.util import current_time


def travel(vehicle, new_requests, graph, max_capacity=2, max_delay=600, debug=False):
    """
    Compute the shortest feasible trip sequence for a vehicle and a set of requests,
    considering all permutations of stops, including flexible pickup positions for new requests.

    Parameters:
    - vehicle: dict, vehicle attributes (current position, trip set, passengers).
    - requests: list of dicts, new requests to validate.
    - graph: networkx.Graph, road network.
    - max_delay: int, maximum allowable delay time in seconds.

    Returns:
    - tuple: (valid_trip, shortest_trip_cost)
        - valid_trip: list of dicts, the shortest valid trip sequence.
        - shortest_trip_cost: float, the total travel cost for the trip.
    """

    if not new_requests and not vehicle["trip_set"]:
        # No requests to process
        return None, float("inf")

    if not new_requests:
        # Only handle drop-offs for existing passengers
        return vehicle["trip_set"], 0  # Assuming no additional cost

    # Prepare stops: drop-offs for existing passengers and both pickup/drop-off for new requests
    stops = [{"type": "dropoff", "node": r["d_r"], "request": r} for r in vehicle["trip_set"]] + \
            [{"type": "pickup", "node": r["o_r"], "request": r} for r in new_requests] + \
            [{"type": "dropoff", "node": r["d_r"], "request": r} for r in new_requests]

    shortest_trip = None
    shortest_trip_cost = float("inf")

    # Helper function to validate pickup-before-dropoff
    def is_valid_permutation(permutation):
        """
        Validate a stop permutation considering pickup-before-dropoff constraints and additional rules
        for when drop-offs must occur before pickups of new requests.

        Parameters:
        - permutation: list of stops (pickup/dropoff)
        Returns:
        - bool: True if the permutation satisfies all constraints, False otherwise.
        """
        if debug:
            print(f'evaluating permutation: {permutation}')

        # Track pickup and drop-off statuses
        pickup_seen = set()
        dropoffs_made = 0
        pickups_made = 0

        # vehicle is empty
        if not vehicle["trip_set"]:
            for stop in permutation:
                if stop["type"] == "dropoff":
                    # Ensure drop-off happens only after its pickup
                    if stop["request"]["id"] not in pickup_seen:
                        if debug:
                            print(f'\t\t\tdropped off before pickup at stop: {stop}')
                        return False
                    dropoffs_made += 1

                elif stop["type"] == "pickup":
                    # Allow pickup since no other passengers are in the vehicle
                    pickup_seen.add(stop["request"]["id"])
                    pickups_made += 1

                    # Ensure pickups do not exceed current drop-offs
                    if pickups_made > dropoffs_made + max_capacity:
                        if debug:
                            print(f'\t\t\tpick up made exceeded maximum: {stop}')
                        return False

        # vehicle is not empty
        else:
            dropoffs_made = 0
            pickups_made =  len(vehicle["trip_set"])
            pickup_seen.update({r["id"] for r in vehicle["trip_set"]})
            min_dropoffs_before_new_pickups = max(0, len(new_requests) - max_capacity + len(vehicle["trip_set"]))

            for stop in permutation:
                if stop["type"] == "dropoff":
                    dropoffs_made += 1

                    # Ensure drop-off happens only after its pickup
                    if stop["request"]["id"] not in pickup_seen:
                        if debug:
                            print(f'\t\t\tdrop-off made before pickup: {stop}')
                        return False

                elif stop["type"] == "pickup":
                    # Enforce rule: New pickups can only occur after required drop-offs
                    if dropoffs_made < min_dropoffs_before_new_pickups:
                        if debug:
                            print(f'\t\t\t pick up can be made after required pick up: pu-made - {pickups_made}, do req: {min_dropoffs_before_new_pickups}, stop: {stop}')
                        return False

                    pickup_seen.add(stop["request"]["id"])
                    pickups_made += 1

                    # Ensure pickups do not exceed current drop-offs + existing trip capacity
                    if pickups_made > dropoffs_made + len(vehicle["trip_set"]):
                        if debug:
                            print(f'\t\t\t vehicle passenger going below 0 - stop: {stop}')
                        return False

        return True

        # # If the vehicle has no initial passengers, no drop-offs are required for them
        # # Check if the vehicle has existing passengers
        # if vehicle["trip_set"]:
        #     # Enforce drop-off requirements only for existing passengers
        #     min_dropoffs_before_new_pickups = max(0, len(new_requests) - max_capacity + len(vehicle["trip_set"]))
        #     # min_dropoffs_before_new_pickups     = max(0, len(new_requests) - max(0, max_capacity - len(vehicle["trip_set"])))
        # else:
        #     # No initial passengers; no drop-off requirements
        #     min_dropoffs_before_new_pickups = 0
        #
        # for stop in permutation:
        #     if stop["type"] == "dropoff":
        #         dropoffs_made += 1
        #
        #     if stop["type"] == "pickup" and stop["request"]["id"] in new_request_ids:
        #         # Enforce rule: New pickups can only occur after required drop-offs
        #         if dropoffs_made < min_dropoffs_before_new_pickups:
        #             print('false1')
        #             return False
        #
        #         # Count pickups for new requests
        #         new_pickups_made += 1
        #
        #         # Allow pickups if the vehicle is empty or the new pickups don't exceed current drop-offs - Ensure no more new pickups occur than allowed before a drop-off
        #         if new_pickups_made > dropoffs_made + len(vehicle["trip_set"]):
        #             print('false2')
        #             return False
        #
        #     # Ensure all pickups happen before their respective drop-offs
        #     # for stop in permutation:
        #     if stop["type"] == "pickup" and stop["request"]["id"] in new_request_ids:
        #         pickup_seen.add(stop["request"]["id"])
        #
        #     elif stop["type"] == "dropoff" and stop["request"]["id"] in new_request_ids:
        #         if stop["request"]["id"] not in pickup_seen:
        #             print('false3')
        #             return False
        #
        # return True

    for stop_order in filter(is_valid_permutation, permutations(stops)):
        if debug:
            for stop in stop_order:
                print(f'\t {stop["request"]["id"]}-{stop["type"]}')

        current_time = vehicle["t_v"]
        current_position = vehicle["q_v"]
        valid = True
        total_cost = 0

        for stop in stop_order:
            if debug:
                print(f'\t {stop["request"]["id"]}-{stop["type"]}')
            # Travel to the next stop
            try:
                travel_time = nx.shortest_path_length(
                    graph, source=current_position, target=stop["node"], weight="travel_time"
                )
            except nx.NetworkXNoPath:
                valid = False
                break

            current_time += travel_time
            total_cost += travel_time

            # Ensure pickup is valid (max waiting time)
            if stop["type"] == "pickup" and current_time > stop["request"]["t_r^pl"]:
                valid = False
                break

            elif stop["type"] == "dropoff":
                # Ensure drop-off is valid (max delay)
                t_r_star = stop["request"]["t_r^*"]
                if current_time > t_r_star + max_delay:
                    valid = False
                    break

            # Update current position
            current_position = stop["node"]

        if valid and total_cost < shortest_trip_cost:
            shortest_trip = stop_order
            shortest_trip_cost = total_cost

    if shortest_trip:
        return [stop for stop in shortest_trip], shortest_trip_cost
    else:
        return None, float("inf")  # No valid trip found

def generate_rv_graph(graph, vehicles, requests, current_time, max_capacity=2, max_delay=600, prune_edges=False, top_k=30, debug=False):
    """
    Generate an RV graph with Request-Request (RR) and Vehicle-Request (VR) edges.

    Parameters:
    - graph: networkx.Graph, the road network.
    - vehicles: list of dicts, each containing vehicle attributes.
    - requests: list of dicts, each containing request attributes.
    - max_capacity: int, maximum passenger capacity per vehicle.
    - max_delay: int, maximum allowable delay time in seconds.
    - prune_edges: bool, whether to limit edges per node.
    - top_k: int, max edges per node when pruning.

    Returns:
    - rv_graph: networkx.Graph, the RV type graph.
    """

    import heapq

    # Helper function for pruning edges
    def _prune_edges(rv_graph, top_k):
        """
        Prune edges in the RV graph to limit the maximum number of edges per node.

        Parameters:
        - rv_graph: networkx.Graph, the RV type graph.
        - top_k: int, maximum number of edges per node.

        Returns:
        - pruned_graph: networkx.Graph, a new graph with pruned edges.
        """
        # Make a copy of the original graph to avoid modifying in place
        pruned_graph = rv_graph.copy()

        for node in list(pruned_graph.nodes):

            if node.startswith("r"):
                edges = [(u, v, data) for u, v, data in pruned_graph.edges(node, data=True) if
                         data.get("edge_type") == "rr"]

            else:
                edges = [(u, v, data) for u, v, data in pruned_graph.edges(node, data=True) if
                         data.get("edge_type") == "rv"]

            # Check if the number of edges exceeds the limit
            if len(edges) > top_k:
                # Sort edges based on "travel_time" (or other cost metric)
                edges_sorted = sorted(edges, key=lambda e: e[2]["travel_time"])

                # Identify edges to remove (edges beyond the top_k)
                edges_to_remove = edges_sorted[top_k:]

                # Remove excess edges
                for u, v, _ in edges_to_remove:
                    pruned_graph.remove_edge(u, v)

        return pruned_graph

    rv_graph = nx.Graph()  # Create an empty type graph

    # Add request nodes
    for request in requests:
        rv_graph.add_node(request["id"], type="request", **request)

    # Add vehicle nodes
    for vehicle in vehicles:
        rv_graph.add_node(vehicle["id"], type="vehicle", **vehicle)

    print("RV graph - VR edge starting...")
    # Step 1: Vehicle-Request (VR) Edges
    for vehicle in vehicles:
        # Skip vehicles without spare capacity
        if vehicle["passengers"] >= max_capacity:
            if debug:
                print(f'\t skipping {vehicle["id"]} due to max passenger')
            continue

        print(f'Processing {vehicle["id"]}... into travel(vehicle, [request])')

        for request in requests:
            if debug:
                print(f'\t generating vid {vehicle["id"]} - request id: {request["id"]}')
            # Use the `travel` function to validate the trip
            valid_trip, trip_cost = travel(vehicle, [request], graph, max_capacity, max_delay, debug)
            if valid_trip:
                if debug:
                    print(f'\t \t adding valid edge {request["id"]} to {vehicle["id"]}')
                rv_graph.add_edge(vehicle["id"], request["id"], travel_time=trip_cost, stops= valid_trip, edge_type='rv')

    # Step 2: Request-Request (RR) Edges
    for req1, req2 in combinations(requests, 2):
        try:
            if debug:
                print(f'\t req-req edge evaluation at {req1["id"]}-{req2["id"]} - combinatorial comparison')

            max_delay_req1 = req1["t_r^*"] + max_delay
            max_delay_req2 = req2["t_r^*"] + max_delay

            # compare max waiting time
            o1d1 = nx.shortest_path_length(graph, source=req1["o_r"], target=req2["o_r"], weight="travel_time")
            d1o2 = nx.shortest_path_length(graph, source=req1["d_r"], target=req2["o_r"], weight="travel_time")
            o2d2 = nx.shortest_path_length(graph, source=req2["o_r"], target=req2["d_r"], weight="travel_time")
            o1o2 = nx.shortest_path_length(graph, source=req1["o_r"], target=req2["o_r"], weight="travel_time")
            d1d2 = nx.shortest_path_length(graph, source=req1["d_r"], target=req2["d_r"], weight="travel_time")

            if (current_time+o1d1 <= max_delay_req1 and current_time+o1d1+d1o2 <= req2["t_r^pl"] and current_time+o1d1+d1o2+o2d2 <= max_delay_req2) or \
                    (current_time+o1o2 <= req2["t_r^pl"] and current_time+o1o2+o2d2 <= max_delay_req2 and current_time+o1o2+o2d2+d1d2 <= max_delay_req1) or \
                    (current_time+o1o2 <= req2["t_r^pl"] and current_time+o1o2+d1o2 <= max_delay_req1 and current_time+o1o2+d1o2+d1d2 <= max_delay_req2):
                rv_graph.add_edge(req1["id"], req2["id"], travel_time=o1o2, edge_type='rr')

        except nx.NetworkXNoPath:
            continue

    # Prune edges if requested
    if prune_edges:
        rv_graph = _prune_edges(rv_graph, top_k=top_k)

    return rv_graph

def visualize_rv_graph(graph, rv_graph, filepath="rv_graph_visualization.png"):
    """
    Visualize the RV graph with VR (Vehicle-Request) and RR (Request-Request) edges overlaid on the road network,
    and differentiate nodes based on their types (vehicles and requests).

    Parameters:
    - graph: networkx.Graph, the road network.
    - rv_graph: networkx.Graph, the RV graph with edge_type attribute ('rv' or 'rr').
    - filepath: str, file path to save the visualization.

    Returns:
    - None (saves the visualization to the specified file path).
    """
    import matplotlib.pyplot as plt
    import osmnx as ox

    # Base road network visualization
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=5, edge_color="gray", bgcolor="white")

    # Get positions for road network nodes
    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}

    # Plot nodes (vehicles and requests)
    for node, data in rv_graph.nodes(data=True):
        if data["type"] == "vehicle":  # Vehicle nodes
            vehicle_pos = pos.get(data["q_v"])
            if vehicle_pos:
                ax.scatter(vehicle_pos[0], vehicle_pos[1], c="blue", s=50, label="Vehicle" if "Vehicle" not in ax.get_legend_handles_labels()[1] else None)
        elif data["type"] == "request":  # Request nodes
            request_pos = pos.get(data["o_r"])
            if request_pos:
                ax.scatter(request_pos[0], request_pos[1], c="red", s=30, label="Request" if "Request" not in ax.get_legend_handles_labels()[1] else None)

    # Draw edges with different colors based on edge_type
    for u, v, edge_data in rv_graph.edges(data=True):
        # edge_type = edge_data.get("edge_type", "unknown")  # Default to unknown if edge_type is missing
        # Determine roles of nodes based on type attribute
        if rv_graph.nodes[u]["type"] == "vehicle" or rv_graph.nodes[v]["type"] == "vehicle":
            if pos.get(rv_graph.nodes[u].get("q_v")):
                vehicle_pos = pos.get(rv_graph.nodes[u].get("q_v"))
                request_pos = pos.get(rv_graph.nodes[v].get("o_r"))
            else:
                vehicle_pos = pos.get(rv_graph.nodes[v].get("q_v"))
                request_pos = pos.get(rv_graph.nodes[u].get("o_r"))

            ax.plot([vehicle_pos[0], request_pos[0]], [vehicle_pos[1], request_pos[1]],
                    c="blue", linestyle="--", alpha=0.7, linewidth=1,
                    label="VR Edge" if "VR Edge" not in ax.get_legend_handles_labels()[1] else None)

        elif rv_graph.nodes[u]["type"] == "request" and rv_graph.nodes[v]["type"] == "request":
            request_u_pos = pos.get(rv_graph.nodes[u].get("o_r"))
            request_v_pos = pos.get(rv_graph.nodes[v].get("o_r"))

            ax.plot([request_u_pos[0], request_v_pos[0]], [request_u_pos[1], request_v_pos[1]],
                    c="green", linestyle="-", alpha=0.7, linewidth=1, label="RR Edge" if "RR Edge" not in ax.get_legend_handles_labels()[1] else None)

        else:
            print(f'warning: unrecognized edge at u: {rv_graph.nodes[u].get("id")} & v: {rv_graph.nodes[v].get("id")}')

    # Add legend and save the figure
    ax.legend(loc="upper right")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def visualize_rv_graph_with_annotations(graph, rv_graph, option_od = True, option_rv = True, filepath="rv_graph_visualization_with_annotation.png"):
    """
    Visualize the RV graph with VR (Vehicle-Request) and RR (Request-Request) edges overlaid on the road network,
    and differentiate nodes based on their types (vehicles and requests).

    Parameters:
    - graph: networkx.Graph, the road network.
    - rv_graph: networkx.Graph, the RV graph with edge_type attribute ('rv' or 'rr').
    - filepath: str, file path to save the visualization.

    Returns:
    - None (saves the visualization to the specified file path).
    """
    import matplotlib.pyplot as plt
    import osmnx as ox

    # Base road network visualization
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=5, edge_color="gray", bgcolor="white")

    # Get positions for road network nodes
    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}

    # Plot nodes (vehicles and requests)
    for node, data in rv_graph.nodes(data=True):
        if data["type"] == "vehicle":  # Vehicle nodes
            vehicle_pos = pos.get(data["q_v"])
            if vehicle_pos:
                ax.scatter(vehicle_pos[0], vehicle_pos[1], c="blue", s=50, label="Vehicle" if "Vehicle" not in ax.get_legend_handles_labels()[1] else None)
                ax.text(vehicle_pos[0], vehicle_pos[1], data["passengers"], fontsize=9, color="green", ha="right", va="top")

                if option_od:
                    if len(data["trip_set"]) > 0:
                        for req_v in data["trip_set"]:
                            dest_v = pos.get(req_v["d_r"])
                            ax.plot([vehicle_pos[0], dest_v[0]], [vehicle_pos[1], dest_v[1]], c="purple",
                                    linestyle="--", marker=">", alpha=0.7, linewidth=1,
                                    label="Vehicle Route" if "Vehicle Route" not in ax.get_legend_handles_labels()[1] else None)
                            vehicle_pos = dest_v

        elif data["type"] == "request":  # Request nodes
            request_pos = pos.get(data["o_r"])
            request_des = pos.get(data["d_r"])

            if request_pos:
                ax.scatter(request_pos[0], request_pos[1], c="red", s=30, label="Request" if "Request" not in ax.get_legend_handles_labels()[1] else None)
                if option_od:
                    ax.plot([request_pos[0], request_des[0]], [request_pos[1], request_des[1]], c="orange", linestyle="--", marker=">", alpha=0.7, linewidth=1,
                            label="OD Pair" if "OD Pair" not in ax.get_legend_handles_labels()[1] else None)

    if option_rv:
        # Draw edges with different colors based on edge_type
        for u, v, edge_data in rv_graph.edges(data=True):
            # edge_type = edge_data.get("edge_type", "unknown")  # Default to unknown if edge_type is missing
            # Determine roles of nodes based on type attribute
            if rv_graph.nodes[u]["type"] == "vehicle" or rv_graph.nodes[v]["type"] == "vehicle":
                if pos.get(rv_graph.nodes[u].get("q_v")):
                    vehicle_pos = pos.get(rv_graph.nodes[u].get("q_v"))
                    request_pos = pos.get(rv_graph.nodes[v].get("o_r"))
                else:
                    vehicle_pos = pos.get(rv_graph.nodes[v].get("q_v"))
                    request_pos = pos.get(rv_graph.nodes[u].get("o_r"))

                ax.plot([vehicle_pos[0], request_pos[0]], [vehicle_pos[1], request_pos[1]],
                        c="purple", linestyle="--", alpha=0.7, linewidth=1,
                        label="VR Edge" if "VR Edge" not in ax.get_legend_handles_labels()[1] else None)

            elif rv_graph.nodes[u]["type"] == "request" and rv_graph.nodes[v]["type"] == "request":
                request_u_pos = pos.get(rv_graph.nodes[u].get("o_r"))
                request_v_pos = pos.get(rv_graph.nodes[v].get("o_r"))

                request_u_des = pos.get(rv_graph.nodes[u].get("d_r"))
                request_v_des = pos.get(rv_graph.nodes[v].get("d_r"))

                ax.plot([request_u_pos[0], request_v_pos[0]], [request_u_pos[1], request_v_pos[1]],
                        c="green", linestyle="-", alpha=0.7, linewidth=1, label="RR Edge" if "RR Edge" not in ax.get_legend_handles_labels()[1] else None)

            else:
                print(f'warning: unrecognized edge at u: {rv_graph.nodes[u].get("id")} & v: {rv_graph.nodes[v].get("id")}')

    # Add legend and save the figure
    ax.legend(loc="upper right")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


"""
Depracated
"""
# allow dynamic re-routing at RV graph calculation
def is_trip_feasible(graph, trip_set, new_request, max_delay=600, current_time=0):
    """
    Check if adding a new request to the trip set is feasible.

    Parameters:
    - graph: networkx.Graph, the road network
    - trip_set: list of dicts, existing trip set with stops
    - new_request: dict, new request with origin and destination
    - max_delay: int, maximum allowed delay in seconds
    - current_time: int, current vehicle time

    Returns:
    - bool: True if feasible, False otherwise
    """
    # Precompute travel time to new request's origin
    try:
        travel_time_to_pickup = nx.shortest_path_length(
            graph, source=trip_set[0]["stop"], target=new_request["o_r"], weight="travel_time"
        )
    except nx.NetworkXNoPath:
        return False  # If there's no path, the request is not feasible

    # Pre-check: Ensure new request pickup doesn't violate existing passenger constraints
    for request in trip_set:
        max_dropoff_time = request["request"]["t_r^*"] + max_delay
        if current_time + travel_time_to_pickup > max_dropoff_time:
            return False  # Pickup would delay existing passenger drop-offs

    # Simulate adding the new request
    pickup_stop = {"stop": new_request["o_r"], "type": "pickup", "request": new_request}
    dropoff_stop = {"stop": new_request["d_r"], "type": "dropoff", "request": new_request}
    candidate_stops = trip_set + [pickup_stop, dropoff_stop]

    # Evaluate valid insertion points
    feasible = False
    for i in range(len(trip_set) + 1):  # Check all insertion points for pickup
        for j in range(i + 1, len(trip_set) + 2):  # Check all insertion points for dropoff
            stops_order = trip_set[:i] + [pickup_stop] + trip_set[i:j] + [dropoff_stop] + trip_set[j:]
            current_sim_time = current_time
            valid = True

            for k in range(len(stops_order) - 1):
                # Compute travel time to the next stop
                try:
                    travel_time = nx.shortest_path_length(
                        graph, source=stops_order[k]["stop"], target=stops_order[k + 1]["stop"], weight="travel_time"
                    )
                except nx.NetworkXNoPath:
                    valid = False
                    break

                current_sim_time += travel_time

                # Check time feasibility for drop-offs
                if stops_order[k + 1]["type"] == "dropoff":
                    t_r_star = stops_order[k + 1]["request"]["t_r^*"]
                    if current_sim_time > t_r_star + max_delay:
                        valid = False
                        break

            if valid:
                feasible = True
                break

        if feasible:
            break

    return feasible

def generate_rv_graph_v0(graph, vehicles, requests, max_capacity=3, max_delay=600):
    """
    Generate an RV graph with dynamic routing and feasibility checks.

    Parameters:
    - graph: networkx.Graph, the road network
    - vehicles: list of dicts, each containing vehicle attributes
    - requests: list of dicts, each containing request attributes
    - max_capacity: int, maximum passenger capacity per vehicle
    - max_delay: int, maximum allowable delay time in seconds

    Returns:
    - rv_graph: networkx.Graph, type graph of vehicles and requests
    """

    debug = True

    if debug:
        print('generating RV graph')

    rv_graph = nx.Graph()  # Create an empty type graph

    # Add vehicle nodes
    for vehicle in vehicles:
        rv_graph.add_node(vehicle["id"], type="vehicle", **vehicle)

    # Add request nodes
    for request in requests:
        rv_graph.add_node(request["id"], type="request", **request)

    # Add edges based on dynamic routing feasibility
    for vehicle in vehicles:
        if debug:
            print(f'checking vehicle {vehicle["id"]}')
        for request in requests:
            if debug:
                print(f'checking request {request["id"]}')
            # Check if the vehicle can reach the request's origin on time
            try:
                travel_time_to_origin = nx.shortest_path_length(
                    graph, source=vehicle["q_v"], target=request["o_r"], weight="travel_time"
                )
                if travel_time_to_origin > (request["t_r^pl"] - request["t_r^r"]):
                    continue  # Cannot reach the request's origin on time

                # Check capacity
                if vehicle["passengers"] + 1 > max_capacity:
                    continue  # Exceeds capacity

                # Check feasibility of adding the request to the trip set
                trip_set = [{"stop": req["o_r"], "type": "pickup", "request": req}
                            for req in vehicle["trip_set"]] + \
                           [{"stop": req["d_r"], "type": "dropoff", "request": req}
                            for req in vehicle["trip_set"]]

                if vehicle["passengers"] > 0 and (not is_trip_feasible(graph, trip_set, request, max_delay, vehicle["t_v"])):
                    continue  # Adding the request is not feasible

                # Add edge if all constraints are satisfied
                rv_graph.add_edge(vehicle["id"], request["id"], travel_time=travel_time_to_origin)
            except nx.NetworkXNoPath:
                # Skip if no path exists
                continue

    return rv_graph

def visualize_rv_graph_with_annotations_v0(graph, rv_graph, vehicles, requests, filepath="rv_graph_annotated.png"):
    """
    Visualize the RV graph with VR (Vehicle-Request) and RR (Request-Request) edges overlaid on the road network,
    and annotate each node with its ID (vehicle or request).

    Parameters:
    - graph: networkx.Graph, the road network.
    - rv_graph: networkx.Graph, the RV graph (type).
    - vehicles: list of dicts, vehicle nodes with attributes.
    - requests: list of dicts, request nodes with attributes.
    - filepath: str, file path to save the visualization.

    Returns:
    - None (saves the visualization to the specified file path).
    """
    import matplotlib.pyplot as plt
    import osmnx as ox

    # Base road network visualization
    fig, ax = ox.plot_graph(graph, show=False, close=False, node_size=5, edge_color="gray", bgcolor="white")

    # Get positions for road network nodes
    pos = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}

    # Plot vehicles with annotations
    for vehicle in vehicles:
        vehicle_pos = pos[vehicle["q_v"]]
        ax.scatter(vehicle_pos[0], vehicle_pos[1], c="blue", s=50, label="Vehicle" if "Vehicle" not in ax.get_legend_handles_labels()[1] else None)
        ax.text(vehicle_pos[0], vehicle_pos[1], vehicle["id"], fontsize=8, color="blue", ha="right")

    # Plot requests with annotations
    for request in requests:
        request_pos = pos[request["o_r"]]
        ax.scatter(request_pos[0], request_pos[1], c="red", s=30, label="Request" if "Request" not in ax.get_legend_handles_labels()[1] else None)
        ax.text(request_pos[0], request_pos[1], request["id"], fontsize=8, color="red", ha="right")

    # Draw VR and RR edges
    for u, v, edge_data in rv_graph.edges(data=True):
        if u.startswith("v") and v.startswith("r"):  # VR edge
            vehicle = next((veh for veh in vehicles if veh["id"] == u), None)
            request = next((req for req in requests if req["id"] == v), None)
            if vehicle and request:  # Ensure both nodes exist
                vehicle_pos = pos[vehicle["q_v"]]
                request_pos = pos[request["o_r"]]
                ax.plot([vehicle_pos[0], request_pos[0]], [vehicle_pos[1], request_pos[1]],
                        c="blue", linestyle="--", alpha=0.7, linewidth=1, label="VR Edge" if "VR Edge" not in ax.get_legend_handles_labels()[1] else None)

        elif u.startswith("r") and v.startswith("r"):  # RR edge
            request_u = next((req for req in requests if req["id"] == u), None)
            request_v = next((req for req in requests if req["id"] == v), None)
            if request_u and request_v:  # Ensure both nodes exist
                request_u_pos = pos[request_u["d_r"]]
                request_v_pos = pos[request_v["o_r"]]
                ax.plot([request_u_pos[0], request_v_pos[0]], [request_u_pos[1], request_v_pos[1]],
                        c="green", linestyle="-", alpha=0.7, linewidth=1, label="RR Edge" if "RR Edge" not in ax.get_legend_handles_labels()[1] else None)

    # Add legend and save the figure
    ax.legend(loc="upper left")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)

def generate_rv_graph_v1(graph, vehicles, requests, max_capacity=2, max_delay=600, prune_edges=False, top_k=10):
    """
    Generate an RV graph with dynamic routing and feasibility checks.

    Parameters:
    - graph: networkx.Graph, the road network.
    - vehicles: list of dicts, each containing vehicle attributes.
    - requests: list of dicts, each containing request attributes.
    - max_capacity: int, maximum passenger capacity per vehicle.
    - max_delay: int, maximum allowable delay time in seconds.
    - prune_edges: bool, whether to limit edges per node.
    - top_k: int, max edges per node when pruning.

    Returns:
    - rv_graph: networkx.Graph, the RV type graph.
    """
    rv_graph = nx.Graph()

    # Add request nodes
    for request in requests:
        rv_graph.add_node(request["id"], type="request", **request)

    # Add vehicle nodes
    for vehicle in vehicles:
        rv_graph.add_node(vehicle["id"], type="vehicle", **vehicle)

    # Request-Request (RR) edges
    for r1, r2 in combinations(requests, 2):
        try:
            travel_time_r1_to_r2 = nx.shortest_path_length(graph, source=r1["o_r"], target=r2["o_r"], weight="travel_time")
            if travel_time_r1_to_r2 <= (r2["t_r^pl"] - r1["t_r^r"]):
                # Check feasibility of serving both requests with max delay
                if travel({"q_v": r1["o_r"], "t_v": r1["t_r^r"], "trip_set": []}, [r1, r2], graph, max_delay=max_delay):
                    cost = (r1["t_r^pl"] - r1["t_r^*"]) + (r2["t_r^pl"] - r2["t_r^*"])
                    rv_graph.add_edge(r1["id"], r2["id"], cost=cost)
        except nx.NetworkXNoPath:
            continue

    # Vehicle-Request (VR) edges
    for vehicle in vehicles:
        for request in requests:
            try:
                # Validate the trip using `travel()`, which includes all necessary constraints
                if vehicle["passengers"] < max_capacity and travel(vehicle, [request], graph, max_delay=max_delay):
                    rv_graph.add_edge(vehicle["id"], request["id"])
            except nx.NetworkXNoPath:
                continue

    valid, total_travel_time = travel(vehicle, [request], graph, max_delay=max_delay)
    if vehicle["passengers"] < max_capacity and valid:
        rv_graph.add_edge(vehicle["id"], request["id"], travel_time=total_travel_time)

    # Prune edges if requested
    if prune_edges:
        for node in rv_graph.nodes:
            edges = list(rv_graph.edges(node, data=True))
            if len(edges) > top_k:
                top_edges = sorted(edges, key=lambda e: e[2]["cost"] if "cost" in e[2] else e[2]["travel_time"])[:top_k]
                rv_graph.remove_edges_from(set(edges) - set(top_edges))

    return rv_graph