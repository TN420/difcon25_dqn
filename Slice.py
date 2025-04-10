# Slice.py

import simpy

class Slice:
    def __init__(self, name, ratio, pk, client_weight,
                 delay_tolerance, qos_class, bandwidth_guaranteed,
                 bandwidth_max, capacity_bandwidth, usage_pattern):
        self.name = name
        self.ratio = ratio
        self.pk = pk
        self.client_weight = client_weight
        self.delay_tolerance = delay_tolerance
        self.qos_class = qos_class
        self.bandwidth_guaranteed = bandwidth_guaranteed
        self.bandwidth_max = bandwidth_max
        self.capacity_bandwidth = capacity_bandwidth
        self.usage_pattern = usage_pattern
        self.capacity = simpy.Container(capacity_bandwidth, capacity_bandwidth)
        self.connected_users = 0

    def is_avaliable(self):
        return self.capacity.level > 0

    def get_consumable_share(self):
        available = self.capacity.level
        return min(available, self.bandwidth_max) if self.bandwidth_max else available

    def __str__(self):
        return f'Slice(name={self.name}, qos_class={self.qos_class}, capacity={self.capacity.level}/{self.capacity.capacity})'
