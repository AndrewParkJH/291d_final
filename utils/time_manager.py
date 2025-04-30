from cachetools import LRUCache
import numpy as np

class TravelTimeManager:
    def __init__(self, network, max_cache_size=2000):
        """Manage cached travel times on the given road network."""
        self.network = network
        self.cache = LRUCache(maxsize=max_cache_size)
        # self.cache = {}  # {(from_node, to_node): travel_time}

    def query(self, from_node, to_node):
        key = (from_node, to_node)
        if key in self.cache:
            entry = self.cache[key]
            return (entry['travel_time'],
                    entry['travel_distance'],
                    entry['path'],
                    entry['segment_times'])

        travel_time, travel_distance, path, segment_times = self.network.fast_network.fast_custom_dijkstra(from_node, to_node)

        if not np.isfinite(travel_time):
            return np.inf, np.inf, [], []
            raise Exception(f"Travel time error in nodes ({from_node}, {to_node}): travel time infinity. Possibly no available path.")

        self.cache[key] = {'travel_time': travel_time,
                           'travel_distance': travel_distance,
                           'path': path,
                           'segment_times': segment_times}

        return travel_time, travel_distance, path, segment_times

    def cleanup(self, node):
        """Remove cached times involving a given node (called when a stop is completed)."""
        for key in list(self.cache.keys()):
            if key[0] == node or key[1] == node:
                del self.cache[key]