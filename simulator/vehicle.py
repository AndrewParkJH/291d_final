from utils.time_manager import TravelTimeManager
import random
import simpy
import networkx as nx

T_MAX = 600 # time penalty normalizer

class Vehicle:
    def __init__(self, env, network, vid, max_capacity=10, randomize_position=True, randomize_passengers=False):

        # simulation parameters
        self.env = env
        self.network = network # RoadNetwork class, RoadNetwork.graph == networkx.Graph, the road network
        self.vehicle_id = vid
        self.max_capacity = max_capacity

        # Vehicle state
        self.current_num_pax = 0
        self.current_requests = {}           # list of request dictionary objects
        self.trip_sequence = []              # list of stops in node ids
        self.trip_sequence_request_id = []   # list of stops corresponding to current request index
        self.new_request = None              # currently assigned new request info (if any)
        self.new_request_index = None        # passenger index for the new request
        self.insertion_step = -1             # 0 = inserting pickup, 1 = inserting dropoff
        self.pickup_idx = 0                  # inserted index for pick up location

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

    def set_new_request(self, request, idx):
        self.new_request = request
        self.new_request_index = idx
        self.insertion_step = 0  # start with pickup insertion stage
        self.pickup_idx = 0

    def reset_new_request(self):
        self.new_request = None
        self.new_request_index = None
        self.insertion_step = -1 # start with pickup insertion stage
        self.pickup_idx = 0

    def add_request(self, request):
        """
        Assign a new request to this vehicle (called when a request is dispatched to this vehicle).
        Precomputes relevant travel times and prepares for insertion decisions.
        """
        if self.current_num_pax >= self.max_capacity:
            return False  # cannot accept more passengers

        # Get next available passenger index
        new_idx = self.find_next_request_index()
        self.current_requests[new_idx] = request
        self.current_requests[new_idx]['remaining_time'] = request['deadline'] - self.env.now
        # self.current_num_pax += request['num_passengers']

        # Store the new request info for the insertion process
        self.set_new_request(request, new_idx)

        # Precompute travel times between new request nodes and active route points
        o_node = request["pu_osmid"]  # pickup node
        d_node = request["do_osmid"]  # dropoff node

        active_nodes = [self.current_node] + self.trip_sequence
        # Cache travel times from each active node to the new pickup/dropoff and vice versa

        for node in active_nodes[1:]:
            self.travel_time_mgr.query(node, o_node)
            self.travel_time_mgr.query(o_node, node)

        self.travel_time_mgr.query(o_node, d_node)

        return True

    def apply_action(self, action):
        """
        Execute the insertion action chosen by the agent:
          0 = No-op (do nothing),
          1..N = insert at index (action-1) in the stop sequence.
        Returns a flag (1 if the action was invalid, 0 otherwise).
        """
        # self.last_insertion_step = -1 if self.new_request is None else self.insertion_step # Track which stage (pickup or dropoff insertion) this action is for
        self.last_action = action

        if self.new_request is None:
            # No pending insertion; only no-op is valid
            if action != 0:
                self.is_invalid_action = True

        else: # if self.new_request is not None:
            if action == 0:
                # Agent chose to do nothing despite a pending request (invalid decision)
                self.is_invalid_action = True
            else:
                insert_idx = action - 1  # because 0 is no-op, real indices start from 0
                if insert_idx > len(self.trip_sequence):
                    self.is_invalid_action = True
                    return True # return invalid flag
                    # raise ValueError("insert idx longer than possible trip sequence")

                if self.insertion_step == 0:
                    # **Pickup insertion**: insert the pickup at the chosen position
                    self.trip_sequence.insert(insert_idx, self.new_request['pu_osmid'])
                    self.trip_sequence_request_id.insert(insert_idx, self.new_request_index)
                    self.last_insertion_cost = self.precomputed_insertion_cost[action]
                    self.insertion_step = 1  # now expect dropoff insertion
                    self.pickup_idx = insert_idx
                    # (Keep self.new_request info until dropoff is placed)

                elif self.insertion_step == 1:
                    # **Dropoff insertion**: insert the dropoff for the new request
                    # Enforce dropoff comes after its pickup
                    if insert_idx <= self.pickup_idx:
                        # dropoff was attempted before pickup (corrected, but mark invalid)
                        self.is_invalid_action = True
                        return True

                    self.trip_sequence.insert(insert_idx, self.new_request['do_osmid'])
                    self.trip_sequence_request_id.insert(insert_idx, self.new_request_index)
                    self.last_insertion_cost = self.precomputed_insertion_cost[action]

                    # Insertion complete – clear new_request
                    self.reset_new_request()

        return self.is_invalid_action

    def find_next_request_index(self):
        """
        Find the smallest unused positive index for a new request.
        """
        idx = 1
        while idx in self.current_requests:
            idx += 1
        return idx

    def get_state(self):
        """
        Build the observation for this vehicle:
            [stage_flag, normalized_stop_count, remaining_capacity, time_constraint,
             pickup_or_dropoff_insertion_costs,
             group_sizes_along_trip_sequence, remaining_times_along_trip_sequence]
        """
        # Stage flag: -1 (no request), 0 (pickup insertion pending), 1 (dropoff insertion pending)

        stage                   = self.insertion_step
        normalized_stop_count   = len(self.trip_sequence) / self.max_capacity  # normalized
        remaining_cap           = self.current_num_pax / self.max_capacity  # normalized

        time_constraint         = 0.0 # Time remaining until deadline for new request
        insertion_costs         = [0.0] * (2 * self.max_capacity + 1)  # Prepare insertion cost vectors (fixed maximum lengths for consistency)
        group_size              = [0.0] * (2 * self.max_capacity)
        remaining_time          = [0.0] * (2 * self.max_capacity)

        current_time        = self.env.now

        # group size and remaining time
        for idx, req_id in enumerate(self.trip_sequence_request_id):
            if req_id is None:
                continue
            request = self.current_requests.get(req_id)
            if request:
                group_size[idx] = request['num_passengers'] / self.max_capacity
                remaining_time[idx] = request['remaining_time'] / 3300

        if self.new_request is not None:
            # compute deadline
            deadline = self.new_request["deadline"]
            time_constraint = max(0.0, deadline - current_time) / 3300
            insertion_node = None
            route_len = len(self.trip_sequence)

            # compute insertion cost
            if self.insertion_step == 0:
                insertion_node = self.new_request["pu_osmid"]
            elif self.insertion_step == 1:
                insertion_node = self.new_request["do_osmid"]

            for j in range(self.pickup_idx + 1):
                insertion_costs[j] = 1.0 # max penalty if insertion of drop off before pick up

            for j in range(self.pickup_idx + 1, route_len + 1):
                # Define prev_node and next_node around position j
                prev_node = self.current_node if j == 0 else self.trip_sequence[j - 1]
                next_node = None
                if j < route_len:
                    next_node = self.trip_sequence[j]

                if prev_node == self.new_request["do_osmid"] and next_node == self.new_request["pu_osmid"]:
                    insertion_costs[j] = 1.0

                else:
                    # Compute baseline travel time (prev -> next without new stop)
                    if next_node is None:
                        base_time = 0.0
                    else:
                        base_time,_,_,_  = self.travel_time_mgr.query(prev_node, next_node)

                    # Compute travel time via new pickup
                    detour_time,_,_,_  = self.travel_time_mgr.query(prev_node, insertion_node)
                    if next_node is not None:
                        next_detour_time,_,_,_ = self.travel_time_mgr.query(insertion_node, next_node)
                        detour_time += next_detour_time
                    # Insertion cost = detour_time - base_time
                    insertion_costs[j] = (detour_time - base_time) / T_MAX # normalize

        self.precomputed_insertion_cost = insertion_costs
        # Assemble the state vector
        state_vec = ([stage, normalized_stop_count, remaining_cap, time_constraint]
                     + insertion_costs + group_size + remaining_time)
        return state_vec

    def remove_request(self, request_index):
        """
        Remove a passenger from the vehicle after completing their dropoff.
        Cleans up cached times for that passenger's origin/destination.
        """
        if request_index in self.current_requests:
            info = self.current_requests.pop(request_index)
            self.current_num_pax = max(0, self.current_num_pax - info["num_passengers"])

            # Clean up travel times involving this passenger's nodes
            # o_node = info["o_r"]
            # d_node = info["d_r"]
            # self.travel_time_mgr.cleanup(o_node)
            # self.travel_time_mgr.cleanup(d_node)

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
            reward -= (5.0 + 0.01*current_late_time)

        # 5. traversing progress reward
        # remaining_segment_time = sum(self.current_segment_times)
        # reward -= 0.01 * remaining_segment_time  # smaller total remaining time = better

        # 6. Insertion cost penalty
        reward -= 2.0 * self.last_insertion_cost  # insertion cost penalty (smaller the better)

        # 2. **Insertion delay penalty** (how much the last request was delayed by routing decisions)
        reward -= 0.01 * self.total_late_time


        # Reward tracking variables
        if self.insertion_step == -1: # meaning no new request or successful insertion after drop off
            self.total_pickups = 0  # total number of passengers picked up in this epoch
            self.total_dropoffs = 0  # total number of passengers dropped off in this epoch
            self.total_late_time = 0.0  # total seconds late across all passengers
            self.last_insertion_cost = 0.0
            self.last_action = None

        return reward

    def update_state(self):
        """
                Move the vehicle to its next stop in the trip sequence.
                If the next stop is a passenger dropoff, award dropoff reward.
                Only assign rewards upon actual dropoffs (and pickups) – moving without an event gives no reward.
                """

        if self.is_invalid_action:
            """
            1) No new request but insertion was tried ==> do nothing
            2) Agent chose to do nothing despite a pending request ==> remove request
            3) Wrong insertion ==> remove request
            """

            if self.new_request is None: # case 1
                return
            else: # case 2 and 3
                self._handle_invalid_action()  # (inside here, call reset_new_request)

        if not self.trip_sequence:
            return  # no stops to go to

        # Get the next stop and associated request
        next_node = self.trip_sequence[0]

        if self.traversal_process is not None and self.traversal_process.is_alive:
            if self.next_node == next_node:
                return
            else:
                self.traversal_process.interrupt()  # kill old traversal if exists

        # start traversal
        self.next_node = next_node
        _, _, route, segment_times = self.travel_time_mgr.query(self.current_node, self.next_node)
        self.current_trajectory = route
        self.current_segment_times = segment_times
        self.traversal_process = self.env.process(self.traverse_trajectory())

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

    def traverse_trajectory(self):
        """
        Traverse node-by-node along current_trajectory.
        """
        try:
            while len(self.current_trajectory) > 1:
                if len(self.current_segment_times) == 0:
                    raise Exception("segment time length zero: edge case detected")
                next_node = self.current_trajectory.pop(1)  # 1 because 0 is current position
                segment_travel_time = self.current_segment_times.pop(0)

                # Move to next node
                yield self.env.timeout(segment_travel_time)

                # Move to next node
                self.current_node = next_node

                # Update internal timers
                self.update_vehicle_state(segment_travel_time)

            # At destination (final node of trajectory)
            self.arrive_at_stop()


        except (simpy.Interrupt, GeneratorExit):
            # If interrupted (new action), do nothing here
            pass

    def arrive_at_stop(self):
        """
        Handle arrival at pickup/dropoff stop.
        """

        # add remaining time to deadline updates

        final_node = self.trip_sequence.pop(0)
        req_id = self.trip_sequence_request_id.pop(0)

        # Reset reward counters
        if req_id is not None:
            request = self.current_requests.get(req_id)
            if request:
                if final_node == request['pu_osmid']:
                    self.current_num_pax += request['num_passengers'] # add passenger if origin is reached
                    self.total_pickups += request['num_passengers'] # Pickup event

                elif final_node == request['do_osmid']:
                    self.total_dropoffs += request['num_passengers'] # Dropoff event

                    if request['remaining_time'] < 0:
                        self.total_late_time += abs(request['remaining_time']) * request['num_passengers']

                    self.remove_request(req_id)

    def update_vehicle_state(self, delta_time):
        """
        Update internal state after moving: decrease remaining time of all requests.
        """
        for req_id, request in self.current_requests.items():
            request['remaining_time'] -= delta_time

    def _initialize_position(self, pos=(-122.41989513021899, 37.792216322686436), randomize=True):
        if randomize:
            self.current_node = random.choice(list(self.network.graph.nodes)) # assign random position
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
