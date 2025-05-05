from utils.time_manager import TravelTimeManager
import random
import simpy
import networkx as nx

T_MAX = 600  # time penalty normalizer


class Vehicle:
    def __init__(self, env, network, vid, max_capacity=10, randomize_position=True, randomize_passengers=False):

        # simulation parameters
        self.env = env
        self.network = network  # RoadNetwork class, RoadNetwork.graph == networkx.Graph, the road network
        self.vehicle_id = vid
        self.max_capacity = max_capacity

        # Vehicle state
        self.current_num_pax = 0
        self.current_requests = {}  # list of request dictionary objects
        self.trip_sequence = []  # list of stops in node ids
        self.max_trip_sequence = self.max_capacity*2
        self.new_request = None  # currently assigned new request info (if any)
        self.new_request_index = None  # passenger index for the new request

        # For graph traversing logic
        self.current_trajectory = []
        self.current_segment_times = []
        self.traversal_process = None

        # Travel time caching manager
        self.travel_time_mgr = TravelTimeManager(network)

        # run initialization
        self.current_pos = None
        self.current_node = None
        self.next_pos = None
        self.next_node = None
        self._initialize_position(randomize=randomize_position)
        self._initialize_current_passengers(randomize=randomize_passengers)

        # Reward tracking variables for RL
        self.is_invalid_action = False
        self.invalid_action_counter = 0
        self.total_pickups = 0  # total number of passengers picked up in this epoch
        self.total_dropoffs = 0  # total number of passengers dropped off in this epoch
        self.total_successful_dropoffs = 0 # total number of passengers dropped off in this epoch within time limit
        self.total_late_time = 0.0  # total seconds late across all passengers
        self.last_action = None
        self.last_insertion_cost = 0
        self.precomputed_insertion_cost = None

        # initial guide to decrease the action space to encourage smaller actions at first
        self.deg_wrong_o_1 = 0
        self.deg_wrong_o_2 = 0
        self.deg_wrong_o_3 = 0
        self.deg_wrong_o_4 = 0

        self.deg_wrong_d_1 = 0
        self.deg_wrong_d_2 = 0
        self.deg_wrong_d_3 = 0
        self.deg_wrong_d_4 = 0


    def add_request(self, request):
        """
        Assign a new request to this vehicle (called when a request is dispatched to this vehicle).
        Precomputes relevant travel times and prepares for insertion decisions.
        """
        if self.current_num_pax >= self.max_capacity:
            return False  # cannot accept more passengers
        # self.current_num_pax += request['num_passengers']

        # Store the new request info for the insertion process
        self.new_request = request
        self.new_request['remaining_time'] = request['deadline'] - self.env.now

        return True

    def insert_request(self, action):
        """
        Execute the insertion action chosen by the agent:
        :param action: action[1,2], index1: insertion index of pickup point, index2: insertion index of dropoff point
        """
        """
        insertion is valid only if current_capacity <= max_capacity - 1 (assume this is checked and true)
        0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18  19  20  21  22  23
        11 12 21 22 31 32 41 42 51 52 61 62 71 72 81 82 91 92 101 102 111 112 

        action space ==> 
            pick up [0~22] ==> [0]+[1-23] (0: do nothing)
            drop off [1~23] ==> [1]+[2-24] (1: do nothing)
        """

        self.is_invalid_action = False
        self.invalid_action_counter = 0
        self.deg_wrong_o_1 = 0
        self.deg_wrong_o_2 = 0
        self.deg_wrong_o_3 = 0
        self.deg_wrong_o_4 = 0
        self.deg_wrong_d_1 = 0
        self.deg_wrong_d_2 = 0
        self.deg_wrong_d_3 = 0
        self.deg_wrong_d_4 = 0

        insert_o = action[0]
        insert_d = action[1]

        self.last_action = action

        # check if no request is present
        if self.new_request is None:
            if insert_o != 0 or insert_d != 0:
                self.invalid_action_counter += 1
                self.is_invalid_action = True
                self.deg_wrong_o_1 = insert_o/self.max_trip_sequence
                self.deg_wrong_d_1 = insert_d/self.max_trip_sequence

        else:  # if new request is present
            if insert_o == 0 or insert_d == 0:
                self.invalid_action_counter += 1
                self.is_invalid_action = True

            else:
                # because 0 is no-op, real indices start from 0
                ins_o_idx = insert_o-1
                ins_d_idx = insert_d

                # check insertion validity: 0 <= pick up insertion index < drop off insertion index <= current trip length
                if not ins_o_idx < ins_d_idx:
                    self.is_invalid_action = True
                    self.invalid_action_counter += 1
                    self.deg_wrong_d_3 = (ins_o_idx - ins_d_idx)/self.max_trip_sequence

                if not ins_d_idx <= len(self.trip_sequence)+1:
                    self.is_invalid_action = True
                    self.invalid_action_counter += 1
                    self.deg_wrong_d_4 = (ins_d_idx - len(self.trip_sequence)+1)/self.max_trip_sequence

                elif ins_o_idx < ins_d_idx:
                    self.trip_sequence.insert(ins_o_idx, {'request_id': self.new_request['request_id'],
                                                          'node_id': self.new_request['pu_osmid'],
                                                          'stage': 'pickup'})

                    self.trip_sequence.insert(ins_d_idx, {'request_id': self.new_request['request_id'],
                                                          'node_id': self.new_request['do_osmid'],
                                                          'stage': 'dropoff'})

                    # update current request after successful insertion
                    request_id = self.new_request['request_id']
                    self.current_requests[request_id] = self.new_request.copy()

        # reset new request after each insertion epoch
        self.new_request = None

        return self.is_invalid_action

    def get_state(self, time_normalizer):
        """
        Build the observation for this vehicle:
            [stage, invalid_action_counter, normalized_stop_count, normalized_remaining_cap, time_constraint
                 + dist_from_trip_seq_o + dist_from_trip_seq_d + group_size + remaining_time]
        """
        # Stage flag: 0 (no request), 1 (new request present)

        stage                   = 0 if self.new_request is None else 1
        invalid_action_counter  = self.invalid_action_counter
        stop_count              = len(self.trip_sequence)  # trip sequence length
        remaining_cap           = self.current_num_pax  # remaining capacity

        time_constraint         = 0.0  # Time remaining until deadline for the new request

        # Prepare insertion cost vectors (fixed maximum lengths for consistency)
        max_sequence_length     = 2 * self.max_capacity
        group_size              = [0.0] * max_sequence_length
        remaining_time          = [0.0] * max_sequence_length
        invalid_flag            = 1 if self.is_invalid_action else 0

        # if new request, cost of not doing anything = 1
        if self.new_request is None: # stage == 0
            if stage != 0:
                raise Exception('stage not 0 event though new request is present')
            insertion_cost_o_1  = [0.0]
            insertion_cost_d_1  = [0.0]
            insertion_cost_o_2  = [1.0] * (max_sequence_length - 1)
            insertion_cost_d_2  = [1.0] * (max_sequence_length - 1)
            insertion_cost_o_3  = []
            insertion_cost_d_3  = []
            insertion_cost_o    = insertion_cost_o_1+insertion_cost_o_2
            insertion_cost_d    = insertion_cost_d_1+insertion_cost_d_2

        else: # stage == 1
            if stage != 1:
                raise Exception('stage not 1 event though new request is present')
            insertion_cost_o_1 = [1.0] # should not take no action if new request is present
            insertion_cost_d_1 = [1.0] # should not take no action if new request is present

            # then, add origin cost (max allowed position for origin is len(trip_sequence)+1 = stop_count+1
            # len(insertion_cost_o_1) = 1, so len(insertion_cost_o_1 + insertion_cost_o_2) = 1 + len(trip_sequence)
            insertion_cost_o_2 = [0.0] * stop_count
            insertion_cost_d_2 = [0.0] * stop_count

            insertion_cost_o_3 = [1.0] * (max_sequence_length - (stop_count + 1))
            insertion_cost_d_3 = [1.0] * (max_sequence_length - (stop_count + 1))

            if len(insertion_cost_o_1+insertion_cost_o_2+insertion_cost_o_3) != max_sequence_length:
                print('debug')

            if self.new_request is not None:
                # compute deadline
                current_time = self.env.now
                deadline = self.new_request["deadline"]
                time_constraint = (deadline - current_time)

        # group size and remaining time
        for idx, seq in enumerate(self.trip_sequence):
            # Skip if we've exceeded the maximum sequence length
            if idx >= max_sequence_length-2: # max idx is 17 (only new passengers can be added if trip sequence is <= 18 (for 10 pax vehicle)
                print(f"Warning: Trip sequence length ({len(self.trip_sequence)}) exceeds maximum capacity ({max_sequence_length})")
                break
                
            request_id = seq['request_id']
            request = self.current_requests.get(request_id)
            if request:
                group_size[idx] = request['num_passengers']
                remaining_time[idx] = request['remaining_time']

                if self.new_request is not None:
                    # compute insertion cost
                    o_node = self.new_request["pu_osmid"]
                    d_node = self.new_request["do_osmid"]

                    node_id = self.trip_sequence[idx]['node_id']
                    insertion_cost_o_2[idx] = self.network.get_euclidean_distance(node_id, o_node)
                    insertion_cost_d_2[idx] = self.network.get_euclidean_distance(node_id, d_node)

        insertion_cost_o = insertion_cost_o_1 + insertion_cost_o_2 + insertion_cost_o_3
        insertion_cost_d = insertion_cost_d_1 + insertion_cost_d_2 + insertion_cost_d_3

        # normalization
        invalid_action_counter = invalid_action_counter/4
        max_distance = max(insertion_cost_o + insertion_cost_d)
        normalized_stop_count = stop_count / max_sequence_length  # # trip sequence normalized
        normalized_remaining_cap = remaining_cap / self.max_capacity  # remaining capacity normalized
        time_constraint = time_constraint/time_normalizer  # Time remaining until deadline for the new request

        # Avoid division by zero
        if max_distance > 0:
            insertion_cost_o = [x / max_distance for x in insertion_cost_o]
            insertion_cost_d = [x / max_distance for x in insertion_cost_d]

        group_size = [size/self.max_capacity for size in group_size]
        remaining_time = [t / time_normalizer for t in remaining_time]

        # add length checker
        if len(insertion_cost_o) != max_sequence_length or len(insertion_cost_d) != max_sequence_length:
            raise Exception("!check insertion cost vector length")

        state_vec = ([stage, invalid_action_counter, normalized_stop_count, normalized_remaining_cap, time_constraint]
                     + insertion_cost_o + insertion_cost_d + group_size + remaining_time)

        return state_vec

    def remove_request(self, request_id):
        """
        Remove a passenger from the vehicle after completing their dropoff.
        Cleans up cached times for that passenger's origin/destination.
        """
        info = self.current_requests.pop(request_id)
        self.current_num_pax = max(0, self.current_num_pax - info["num_passengers"])

        # clean up any incomplete trip sequence
        for ind, trip in enumerate(self.trip_sequence):
            if trip['request_id']==request_id:
                self.trip_sequence.pop(ind)

    def update(self):
        """
        Move the vehicle to its next stop in the trip sequence.
        If the next stop is a passenger dropoff, award dropoff reward.
        Only assign rewards upon actual dropoffs (and pickups) – moving without an event gives no reward.
        """
        if not self.trip_sequence:
            return  # no stops to go to

        # Get the next stop and associated request
        next_request = self.trip_sequence[0]
        next_node    = next_request['node_id']

        # If already moving to the correct node, continue
        if self.traversal_process is not None and self.traversal_process.is_alive:
            if self.next_node == next_node:
                return
            else:
                self.traversal_process.interrupt()  # kill old traversal if exists

        # start traversal
        self.next_node = next_node
        _, _, route, segment_times = self.travel_time_mgr.query(self.current_node, self.next_node)

        # Validate route and segment times
        if not route or not segment_times:
            if self.current_node != self.next_node:
                print(f"Warning: Invalid route or segment times for {self.current_node} -> {self.next_node}, recomputing route and time")
                _, _, route, segment_times = self.travel_time_mgr.query(self.current_node, self.next_node)

        if len(route) - 1 != len(segment_times):
            print(f"Warning: {len(route) - 1} edges but {len(segment_times)} segment times")
            _, _, route, segment_times = self.travel_time_mgr.query(self.current_node, self.next_node)

        self.current_trajectory = route[1:]
        self.current_segment_times = segment_times

        debug = False
        if debug:
            # Log route information for debugging
            print(f"Vehicle {self.vehicle_id} route update:")
            print(f"  Current node: {self.current_node}")
            print(f"  Next node: {self.next_node}")
            print(f"  Route length: {len(route)}")
            print(f"  Segment times: {len(segment_times)}")
            print(f"  Trip sequence length: {len(self.trip_sequence)}")

        self.traversal_process = self.env.process(self.traverse_trajectory())

    def traverse_trajectory(self):
        """
        Traverse node-by-node along current_trajectory with smooth acceleration and deceleration.
        """
        try:
            while len(self.current_trajectory) > 0:
                if len(self.current_segment_times) == 0:
                    raise Exception("segment time length zero: edge case detected")

                next_node = self.current_trajectory.pop(0)
                segment_travel_time = self.current_segment_times.pop(0)

                # Get start and end positions
                start_pos = self.network.get_node_coordinate(self.current_node)
                end_pos = self.network.get_node_coordinate(next_node)

                visual = False

                if visual:
                    """
                    --- visualization
                    """
                    # Calculate distance and speed
                    distance = self.network.get_euclidean_distance(self.current_node, next_node)
                    avg_speed = distance / segment_travel_time if segment_travel_time > 0 else 0

                    self.current_node = next_node

                    # Movement parameters
                    steps = int(segment_travel_time / 1.0)

                    for step in range(steps):
                        progress = (step + 1) / steps  # Linear progress from 0 → 1
                        intermediate_lon = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                        intermediate_lat = start_pos[1] + (end_pos[1] - start_pos[1]) * progress
                        self.current_pos = (intermediate_lon, intermediate_lat)
                        yield self.env.timeout(1.0)

                    """
                    --- visualization
                    """

                else:
                    # Move to next node
                    self.current_node = next_node
                    yield self.env.timeout(segment_travel_time)

                # Update node and position
                self.current_pos = end_pos

                # Update internal timers
                self.update_request_wait_time(segment_travel_time)

            # At destination (final node of trajectory)
            self.arrive_at_stop()


        except (simpy.Interrupt, GeneratorExit):
            # If interrupted (new action), do nothing here
            pass

    def update_request_wait_time(self, delta_time):
        for req_id, request in self.current_requests.items():
            request['remaining_time'] -= delta_time

    def arrive_at_stop(self):
        """
        Handle arrival at pickup/dropoff stop.
        """

        # add remaining time to deadline updates

        request = self.trip_sequence.pop(0)
        request_id  = request['request_id']
        node_id     = request['node_id']
        stage       = request['stage'] # pickup or dropoff


        # Reset reward counters
        if request_id is not None:
            request = self.current_requests.get(request_id)
            if request:
                if stage == 'pickup':
                    self.current_num_pax += request['num_passengers']  # add passenger if origin is reached
                    self.total_pickups += request['num_passengers']  # Pickup event

                elif stage == 'dropoff':
                    self.total_dropoffs += request['num_passengers']  # Dropoff event

                    if request['remaining_time'] > 0:
                        self.total_successful_dropoffs += request['num_passengers']

                    if request['remaining_time'] < 0:
                        self.total_late_time += abs(request['remaining_time']) * request['num_passengers']

                    self.remove_request(request_id)

                else:
                    raise Exception('Edge case detected while processing trip sequence')

    def safe_interrupt_if_moving_to(self, bad_nodes):
        """
        If currently traversing toward any node in bad_nodes, safely interrupt the traversal.
        """
        if self.traversal_process is not None and self.traversal_process.is_alive:
            if self.next_node in bad_nodes:
                print(
                    f"[{self.env.now}] Vehicle {self.vehicle_id}: Interrupting traversal to node {self.next_node} due to invalid request.")
                self.traversal_process.interrupt()

    def _initialize_position(self, pos=(-122.41989513021899, 37.792216322686436), randomize=True):
        random.seed(self.vehicle_id)
        if randomize:
            self.current_node = random.choice(list(self.network.graph.nodes))  # assign random position
            self.current_pos = self.network.get_node_coordinate(self.current_node)

        else:
            self.current_pos = pos
            self.current_node = self.network.find_nearest_node_from_coordinate(self.current_pos)

    def _initialize_current_passengers(self, count=0, randomize=True):
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
            self.current_requests.clear()
            self.trip_sequence.clear()


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
                pickup_time = current_time + ((i + 1) * time_offset)  # Offset based on passenger order
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
            "t_r^r": pickup_time - 30,
            "t_r^pl": pickup_time - 30 + omega,
            "t_r^p": pickup_time,  # Pickup time
            "t_r^d": expected_arrival_time,  # Drop-off time
            "t_r^*": expected_arrival_time - 30  # Earliest drop-off time
        })

    return trip_set
