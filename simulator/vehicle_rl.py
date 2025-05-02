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
        self.total_pickups = 0  # total number of passengers picked up in this epoch
        self.total_dropoffs = 0  # total number of passengers dropped off in this epoch
        self.total_late_time = 0.0  # total seconds late across all passengers
        self.last_action = None
        self.last_insertion_cost = 0
        self.precomputed_insertion_cost = None

    def add_request(self, request):
        """
        Assign a new request to this vehicle (called when a request is dispatched to this vehicle).
        Precomputes relevant travel times and prepares for insertion decisions.
        """
        if self.current_num_pax >= self.max_capacity:
            return False  # cannot accept more passengers

        request_id = request['request_id']

        self.current_requests[request_id] = request
        self.current_requests[request_id]['remaining_time'] = request['deadline'] - self.env.now
        # self.current_num_pax += request['num_passengers']

        # Store the new request info for the insertion process
        self.new_request = request

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

        insert_o = action[0]
        insert_d = action[1]

        self.last_action = action

        # check if no request is present
        if self.new_request is None:
            if insert_o != 0 or insert_d != 0:
                self.is_invalid_action = True
                return True

        else:  # if new request is present
            if insert_o == 0 or insert_d == 0:
                self.is_invalid_action = True
                return True

            else:
                # because 0 is no-op, real indices start from 0
                ins_o_idx = insert_o
                ins_d_idx = insert_d+1

                # check insertion validity: 0 <= pick up insertion index < drop off insertion index <= current trip length
                if not ((0 <= ins_o_idx) and (ins_o_idx < ins_d_idx) and (ins_d_idx <= len(self.trip_sequence))):
                    self.is_invalid_action = True
                    return True

                else:
                    self.trip_sequence.insert(ins_o_idx, {'request_id': self.new_request['request_id'],
                                                          'node_id': self.new_request['pu_osmid'],
                                                          'stage': 'pickup'})

                    self.trip_sequence.insert(ins_d_idx, {'request_id': self.new_request['request_id'],
                                                          'node_id': self.new_request['do_osmid'],
                                                          'stage': 'dropoff'})

                    # reset new request after successful insertion
                    self.new_request = None
                    self.is_invalid_action = False
                    return False

    def get_state(self, time_normalizer):
        """
        Build the observation for this vehicle:
            [stage_flag, normalized_stop_count, remaining_capacity, time_constraint,
             pickup_or_dropoff_insertion_costs,
             group_sizes_along_trip_sequence, remaining_times_along_trip_sequence]
        """
        # Stage flag: 0 (no request), 1 (new request present)

        stage                   = 0 if self.new_request is None else 1
        stop_count              = len(self.trip_sequence)  # trip sequence length
        remaining_cap           = self.current_num_pax  # remaining capacity

        time_constraint         = 0.0  # Time remaining until deadline for the new request

        # Prepare insertion cost vectors (fixed maximum lengths for consistency)
        dist_from_trip_seq_o    = [0.0] * (2 * self.max_capacity) # 0 + 1~23
        dist_from_trip_seq_d    = [0.0] * (2 * self.max_capacity) # 1 + 2~24
        group_size              = [0.0] * (2 * self.max_capacity)
        remaining_time          = [0.0] * (2 * self.max_capacity)

        invalid_flag            = 1 if self.is_invalid_action else 0


        if self.new_request is not None:
            # compute deadline
            current_time = self.env.now
            deadline = self.new_request["deadline"]
            time_constraint = (deadline - current_time) / 120 # normalized by 2 minute

        # group size and remaining time
        for idx, seq in enumerate(self.trip_sequence):
            request_id = seq['request_id']
            request = self.current_requests.get(request_id)
            group_size[idx] = request['num_passengers']
            remaining_time[idx] = request['remaining_time']

            if self.new_request is not None:
                # compute insertion cost
                o_node = self.new_request["pu_osmid"]
                d_node = self.new_request["do_osmid"]

                node_id = self.trip_sequence[idx]['node_id']
                dist_from_trip_seq_o[idx] = self.network.get_euclidean_distance(node_id, o_node)
                dist_from_trip_seq_d[idx] = self.network.get_euclidean_distance(node_id, d_node)

        # normalization
        max_distance                = max(dist_from_trip_seq_o + dist_from_trip_seq_d)
        normalized_stop_count       = stop_count / (2 * self.max_capacity)  # # trip sequence normalized
        normalized_remaining_cap    = remaining_cap / self.max_capacity  # remaining capacity normalized
        time_constraint             = time_constraint/time_normalizer  # Time remaining until deadline for the new request

        # Avoid division by zero
        if max_distance > 0:
            dist_from_trip_seq_o    = [x / max_distance for x in dist_from_trip_seq_o]
            dist_from_trip_seq_d    = [x / max_distance for x in dist_from_trip_seq_d]

        group_size                  = [size/self.max_capacity for size in group_size]
        remaining_time              = [t / time_normalizer for t in remaining_time]


        state_vec = ([stage, invalid_flag, normalized_stop_count, normalized_remaining_cap, time_constraint]
                     + dist_from_trip_seq_o + dist_from_trip_seq_d + group_size + remaining_time)
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

    def compute_reward(self):
        """
        TODO:
            1) Penalize number of invalid actions
            2) Reward pickup #pax
            3) Reward drop-off #pax
            4) Penalize current delay amount in current passengers
            apply insertion delay reward (negative)
            4) Penalty for delays: for each request for deadline is missed, apply growing negative reward (multiplied by group size)
            5)
            7) route progress shaping: making progress toward destinations.
            8) intermediate drop off bonus: making progress toward the drop off
        """
        """
        Compute the reward for the vehicle based on the last action and trip events.
        """
        # Initialize reward components
        reward = 0.0

        # 1. **Invalid action penalties** (e.g., wrong insertion or unnecessary action)
        if self.is_invalid_action:
            self.is_invalid_action = False
            reward += -5.0

            bad_insertion_act_degree = self.last_action - len(self.trip_sequence)
            if bad_insertion_act_degree > 0:
                reward -= bad_insertion_act_degree

            return reward

        # 2. **Pickup event reward** (passengers picked up and on board)
        reward += 1.0 * self.total_pickups

        # 3. **Dropoff success reward** (passengers delivered to destination)
        reward += 3.0 * self.total_dropoffs

        # 4. Current delays
        current_late_time = 0.0
        for request in self.current_requests.values():
            if request['remaining_time'] < 0:
                current_late_time += abs(request['remaining_time']) * request['num_passengers']
        if current_late_time > 0:
            reward -= (5.0 + 0.01 * current_late_time)

        # 5. traversing progress reward
        # remaining_segment_time = sum(self.current_segment_times)
        # reward -= 0.01 * remaining_segment_time  # smaller total remaining time = better

        # 6. Insertion cost penalty
        reward -= 2.0 * self.last_insertion_cost  # insertion cost penalty (smaller the better)

        # 2. **Insertion delay penalty** (how much the last request was delayed by routing decisions)
        reward -= 0.01 * self.total_late_time

        # Reward tracking variables
        if self.insertion_step == -1:  # meaning no new request or successful insertion after drop off
            self.total_pickups = 0  # total number of passengers picked up in this epoch
            self.total_dropoffs = 0  # total number of passengers dropped off in this epoch
            self.total_late_time = 0.0  # total seconds late across all passengers
            self.last_insertion_cost = 0.0
            self.last_action = None

        return reward

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

        if self.traversal_process is not None and self.traversal_process.is_alive:
            if self.next_node == next_node:
                return
            else:
                self.traversal_process.interrupt()  # kill old traversal if exists

        # start traversal
        self.next_node = next_node
        _, _, route, segment_times = self.travel_time_mgr.query(self.current_node, self.next_node)
        self.current_trajectory = route[1:]
        self.current_segment_times = segment_times
        self.traversal_process = self.env.process(self.traverse_trajectory())

    def traverse_trajectory(self):
        """
        Traverse node-by-node along current_trajectory.
        """
        try:
            while len(self.current_trajectory) > 0:
                if len(self.current_segment_times) == 0:
                    raise Exception("segment time length zero: edge case detected")

                next_node = self.current_trajectory.pop(0)
                segment_travel_time = self.current_segment_times.pop(0)

                # Move to next node
                yield self.env.timeout(segment_travel_time)

                # Move to next node
                self.current_node = next_node

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

                    if request['remaining_time'] < 0:
                        self.total_late_time += abs(request['remaining_time']) * request['num_passengers']

                    self.remove_request(request_id)

                else:
                    raise Exception('Edge case detected while processing trip sequence')

    def _handle_invalid_action(self):
        if self.new_request is not None:
            bad_nodes = [self.new_request['pu_osmid'], self.new_request['do_osmid']]
            self.safe_interrupt_if_moving_to(bad_nodes)
            self.remove_request(self.new_request_index)
            self._remove_failed_insertion(self.new_request['request_id'])
            self.reset_new_request()

        self.is_invalid_action = False

    def safe_interrupt_if_moving_to(self, bad_nodes):
        """
        If currently traversing toward any node in bad_nodes, safely interrupt the traversal.
        """
        if self.traversal_process is not None and self.traversal_process.is_alive:
            if self.next_node in bad_nodes:
                print(
                    f"[{self.env.now}] Vehicle {self.vehicle_id}: Interrupting traversal to node {self.next_node} due to invalid request.")
                self.traversal_process.interrupt()

    def _remove_failed_insertion(self, request_id):
        """
        Remove the nodes and request IDs related to a failed insertion.
        """
        indices_to_remove = []

        for idx, req_id in enumerate(self.trip_sequence_request_id):
            if req_id == request_id:
                indices_to_remove.append(idx)

        # Remove in reverse order to avoid messing up indexing
        for idx in reversed(indices_to_remove):
            del self.trip_sequence[idx]
            del self.trip_sequence_request_id[idx]


    def _initialize_position(self, pos=(-122.41989513021899, 37.792216322686436), randomize=True):
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
