import random
import networkx as nx

class Vehicle:
    def __init__(self, env, network, vid, max_capacity=10, randomize=True):
        self.env = env
        self.network = network # RoadNetwork class, RoadNetwork.graph == networkx.Graph, the road network

        self.vehicle_id = vid
        self.max_capacity = max_capacity
        self.current_node = 0
        self.next_node = 0
        self.current_pos = (0,0)
        self.next_pos = (0,0)
        self.current_num_pax = 0
        self.trips = [] # list containing dictionary variables
        """
        Trip Set: 
        trip_set.append({
        "id": # Request ID
        "o_r": origin,  # Origin node
        "d_r": destination,  # Destination node
        "t_r^r": pickup_time-30,
        "t_r^pl": pickup_time - 30 + omega, # latest drop off time
        "t_r^p": pickup_time,  # Pickup time
        "t_r^d": expected_arrival_time,  # Drop-off time
        "t_r^*": expected_arrival_time-30  # Earliest drop-off time
        })
        """
        self.current_passengers = []

        # run initialization
        self.initialize_position(randomize=randomize)
        self.initialize_current_passengers(randomize=randomize)


    def initialize_position(self, pos=0, randomize=True):
        if randomize:
            self.current_node = random.choice(list(self.network.graph.nodes)) # assign random position
            self.current_pos = self.network.get_node_coordinate(self.current_node)

        else:
            self.current_pos = pos

    def initialize_current_passengers(self, count=0, randomize=True):
        if randomize:
            self.current_num_pax = random.randint(0, self.max_capacity)
            self.trips = generate_requests_for_vehicle(self.network.graph, self.current_node, self.current_num_pax)
            if self.current_num_pax > 0:
                self.current_node = self.trips[0]['o_r']
                self.next_node = self.trips[0]['d_r']
                self.current_pos = self.network.get_node_coordinate(self.current_node)
                self.next_pos = self.network.get_node_coordinate(self.next_node)
        else:
            self.current_num_pax = count


    def serve_itinerary(self):
        for trip in self.trips:
            self.next_node = trip['d_r']
            self.env.process(self.travel(self.next_node))

    def travel(self, pos):
        self.next_node = pos

        # calculate travel time and travel sequence
        travel_time = self.network.find_shortest_travel_time(self.current_node, self.next_node)

        self.env.timeout(travel_time)

        # log passenger drop off

        # implement process interrupt

    def interrupt_and_pickup_request(self, request):

        return 0



def generate_requests_for_vehicle(graph, vehicle_node, num_passengers, omega=600, max_delay=600):
    """
    Generate a set of requests for a given vehicle and its passengers, ensuring feasibility.

    Parameters:
    - graph: networkx.Graph, road network with precomputed travel times.
    - vehicle_node: int, current position of the vehicle (node ID).
    - num_passengers: int, number of passengers to generate requests for.
    - max_delay: int, maximum delay allowed for arrival at destinations (in seconds).

    Returns:
    - trip_set: list of dicts, each representing a request with pickup/drop-off info.
    """
    nodes = list(graph.nodes)
    trip_set = []
    current_time = 0  # Assume the vehicle is at time 0
    time_offset = -80  # Each passenger picked up at intervals of -80 seconds

    for i in range(num_passengers):
        pickup_time = 0
        expected_arrival_time = 0
        while True:
            # Generate random origin and destination
            origin = vehicle_node if i == 0 else random.choice(nodes)  # First origin is vehicle's current position
            destination = random.choice(nodes)
            while destination == origin:  # Ensure origin and destination are different
                destination = random.choice(nodes)

            try:
                # Calculate travel time from origin to destination
                travel_time = nx.shortest_path_length(
                    graph, source=origin, target=destination, weight="travel_time"
                )
                # Calculate pickup time and drop-off time
                pickup_time = current_time + ((i+1) * time_offset)  # Offset based on passenger order
                expected_arrival_time = pickup_time + travel_time

                # Ensure drop-off time satisfies time constraints
                if expected_arrival_time <= (pickup_time + max_delay):
                    break  # Exit loop if valid request
            except nx.NetworkXNoPath:
                continue  # Retry with a new destination if no path exists

        # Add request to trips set
        trip_set.append({
            "id": f"r_p{i + 1}",  # Request ID
            "o_r": origin,  # Origin node
            "d_r": destination,  # Destination node
            "t_r^r": pickup_time-30,
            "t_r^pl": pickup_time - 30 + omega,
            "t_r^p": pickup_time,  # Pickup time
            "t_r^d": expected_arrival_time,  # Drop-off time
            "t_r^*": expected_arrival_time-30  # Earliest drop-off time
        })

    return trip_set
