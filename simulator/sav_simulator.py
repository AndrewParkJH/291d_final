from ilp_solver import ILP_Solver

DEBUG = True

class ShuttleSim:
    def __init__(self, env, network, run_mode, request_df, end_time=36000):
        self.one = 0
        self.env = env
        self.network = network
        self.run_mode = run_mode #"benchmark for ilp"
        self.accumulation_time = 120
        self.end_time = end_time
        self.request_data = request_df
        self.current_request = []
        self.dispatch_trigger = env.event()

        if self.run_mode == "benchmark":
            self.ilp_solver = ILP_Solver(self.env, self.network, omega=900, max_delay=600)

        else:
            self.ilp_solver = None

    def step(self):
        # request accumulate
        while True:
            print(f"[{self.env.now}] Tick")
            self.request_accumulate()
            yield self.env.timeout(self.accumulation_time)

            trigger = self.dispatch_trigger
            self.dispatch_trigger = self.env.event()
            trigger.succeed()

            # clear accumulated requests after dispatching
            self.current_request.clear()  # add conditional (clear served requests only) later

            if self.env.now >= self.end_time:
                break

    def request_accumulate(self):
        """

        """
        time_now = self.env.now
        time_next = self.env.now + self.accumulation_time
        # accumulate requests here...
        request_df = self.request_data[
            (self.request_data["time_departure"] >= time_now) &
            (self.request_data["time_departure"] < time_next)
            ]

        for _, request in request_df.iterrows():
            self.current_request.append(request.to_dict())

        if DEBUG:
            print(f"\t{self.env.now}: triggered request accumulate for ({time_now, time_next}) -  current request {self.current_request}")


    def trigger_dispatch(self):
        while True:
            print(f"{self.env.now}: Waiting for customer accumulation")
            yield self.dispatch_trigger
            print(f"[{self.env.now}] Tick | {len(self.current_request)} requests")

            # dispatch logic here...
            self.ilp_solver.solve(self.current_request)

            # benchmark: ilp logic
            if self.run_mode == "benchmark":
                print(f"\tbenchmark logic performed")
                # benchmark logic here