from simulator.sav_simulator_rl import ShuttleSim

sim_kwargs = {
    'trip_date': "2019-09-17",
    'simulation_start_time': 7*3600,
    'simulation_end_time': 10*3600,
    'accumulation_time': 120,
    'num_vehicles': 40,
    'randomize_vehicle_position': True,
    'vehicle_capacity': 10
}

def main():

    simulator = ShuttleSim(**sim_kwargs)
    simulator.reset_simulator()

    print("")




if __name__ == "__main__":
    main()