from simulator.road_network import RoadNetworkTester
from utils.time_manager import TravelTimeManager

import osmnx as ox
import time
import random
import numpy as np

DEFAULT_GRAPH_PATH = "./data/road_network/sf_road_network.graphml"

test_number = 1000

def main():
    rn = RoadNetworkTester()
    ttm = TravelTimeManager(rn)

    # Start timing
    print("grenerating random nodes")
    nodes = []
    for i in range(test_number):
        node1 = random.choice(list(rn.graph.nodes))
        node2 = random.choice(list(rn.graph.nodes))
        nodes.append((node1, node2))

    total_time, total_distance, path, segment_times = None, None, None, None

    # print("FRN Speed Test - CRS built in dijkstra")
    # start_time = time.time()
    # for i in range(len(nodes)):
    #     node1 = nodes[i][0]
    #     node2 = nodes[i][1]
    #     total_time, total_distance, _,_ = rn.fast_network.fast_dijkstra(node1, node2)
    #     if i % 10 == 0:
    #         print(i, time.time()-start_time)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"\tElapsed time for {test_number} calls: {elapsed_time:.6f} seconds")
    # print(f"\tAverage time per call: {elapsed_time / test_number:.6f} seconds")
    # print("\t check result: ", total_time, total_distance)

    print("FRN Speed Test - custom dijkstra")
    total_time, path, segment_times = None,None,None
    start_time = time.time()
    for i in range(len(nodes)):
        node1 = nodes[i][0]
        node2 = nodes[i][1]
        total_time, total_distance, _,_ = rn.fast_network.fast_custom_dijkstra(node1, node2)
        if i % (test_number/10) == 0:
            print(i, time.time()-start_time)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\tElapsed time for {test_number} calls: {elapsed_time:.6f} seconds")
    print(f"\tAverage time per call: {elapsed_time/test_number:.6f} seconds")
    print("\t check result: ", total_time, total_distance)

    # print("FRN Speed Test - fast bidrectional Dijkstra")
    # start_time = time.time()
    # for i in range(len(nodes)):
    #     node1 = nodes[i][0]
    #     node2 = nodes[i][1]
    #     total_time, total_distance, _,_  = rn.fast_network.fast_bidirectional_dijkstra(node1, node2)
    #     if i % 10 == 0:
    #         print(i, time.time()-start_time)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # print(f"\tElapsed time for {test_number} calls: {elapsed_time:.6f} seconds")
    # print(f"\tAverage time per call: {elapsed_time / test_number:.6f} seconds")
    # print("\t check result: ", total_time, total_distance)

    print("FRN Speed Test - fast A* search")
    start_time = time.time()
    for i in range(len(nodes)):
        node1 = nodes[i][0]
        node2 = nodes[i][1]
        total_time, total_distance, _,_  = rn.fast_network.fast_astar(node1, node2)
        if i % (test_number/10) == 0:
            print(i, time.time()-start_time)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\tElapsed time for {test_number} calls: {elapsed_time:.6f} seconds")
    print(f"\tAverage time per call: {elapsed_time / test_number:.6f} seconds")
    print("\t check result: ", total_time, total_distance)

    print("FRN Accuracy Test: fast A* vs Custom Dijkstra")

    total_deviation = 0.0
    max_deviation = 0.0

    start_time = time.time()
    for i in range(len(nodes)):
        node1, node2 = nodes[i]

        # Get ground truth travel time
        true_time, true_distance, _, _ = rn.fast_network.fast_custom_dijkstra(node1, node2)

        # Get A* estimated travel time
        astar_time, astar_distance, _, _ = rn.fast_network.fast_astar(node1, node2)

        # Only if path exists in both
        if np.isfinite(true_time) and np.isfinite(astar_time):
            deviation = abs(astar_time - true_time)
            total_deviation += deviation
            if deviation > max_deviation:
                max_deviation = deviation

        if i % (test_number // 10) == 0:
            print(f"{i}/{test_number} completed. Elapsed {time.time() - start_time:.2f} sec")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Final report
    average_deviation = total_deviation / test_number

    print(f"\n✅ Completed {test_number} OD pairs")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    print(f"Average time per call: {elapsed_time / test_number:.6f} seconds")
    print(f"Average deviation in travel_time (seconds): {average_deviation:.6f}")
    print(f"Maximum deviation observed (seconds): {max_deviation:.6f}")


    # print("TTM Speed Test")
    # start_time = time.time()
    # for i in range(len(nodes)):
    #     node1 = nodes[i][0]
    #     node2 = nodes[i][1]
    #     ttm.query(node1, node2)
    #     if i %10 == 0:
    #         print(i, time.time())





if __name__ == "__main__":
    main()