# Stats.py

from sklearn.ensemble import IsolationForest  # Import Isolation Forest
from skopt import gp_minimize  # Import Bayesian optimization
from skopt.space import Real
from statistics import mean  # Add this import
import numpy as np
import random

class Stats:
    def __init__(self, env, base_stations, clients, area):
        self.env = env
        self.base_stations = base_stations
        self.clients = clients
        self.area = area
        #self.graph = graph

        # Stats
        self.total_connected_users_ratio = []
        self.total_used_bw = []
        self.avg_slice_load_ratio = []
        self.avg_slice_client_count = []
        self.coverage_ratio = []
        self.connect_attempt = []
        self.block_count = []
        self.handover_count = []
        self.block_ratio_anomalies = []  # Store anomaly detection results
        self.isolation_forest = IsolationForest(contamination=0.05, random_state=42)  # Initialize Isolation Forest
        self.base_station_capacities = {bs.pk: [] for bs in base_stations}  # Track capacity over time
        self.optimization_space = [Real(0.1, 2.0, name=f"scale_bs_{bs.pk}") for bs in base_stations]
        self.optimization_results = []
        self.block_ratio_upper_threshold = 0.3  # Example upper threshold
        self.block_ratio_lower_threshold = 0.1  # Example lower threshold
        self.capacity_adjustment_factor = 0.1  # Adjust capacity by 10%

        # Q-learning
        self.q_table = {}  # Q-table for state-action values
        self.learning_rate = 0.1  # Learning rate for Q-learning
        self.discount_factor = 0.95  # Discount factor for future rewards
        self.exploration_rate = 1.0  # Exploration rate for epsilon-greedy policy
        self.exploration_decay = 0.995  # Decay rate for exploration
        self.exploration_min = 0.01  # Minimum exploration rate

    def get_stats(self):
        return (
            self.total_connected_users_ratio,
            self.total_used_bw,
            self.avg_slice_client_count,  # Removed avg_slice_load_ratio
            self.coverage_ratio,
            self.block_count,
            self.block_ratio_anomalies,  # Include anomalies in stats
            self.base_station_capacities,  # Include base station capacities in stats
        )

    def collect(self):
        yield self.env.timeout(0.25)
        self.connect_attempt.append(0)
        self.block_count.append(0)
        self.handover_count.append(0)
        while True:
            self.block_count[-1] /= self.connect_attempt[-1] if self.connect_attempt[-1] != 0 else 1
            self.handover_count[-1] /= self.connect_attempt[-1] if self.connect_attempt[-1] != 0 else 1

            # Detect anomalies in block ratio
            if len(self.block_count) > 10:  # Ensure enough data points for training
                block_ratios = [[b] for b in self.block_count[-10:]]  # Use the last 10 block ratios
                predictions = self.isolation_forest.fit_predict(block_ratios)
                self.block_ratio_anomalies.append(predictions[-1])  # Append the latest prediction (-1 for anomaly, 1 for normal)

            self.total_connected_users_ratio.append(self.get_total_connected_users_ratio())
            self.total_used_bw.append(self.get_total_used_bw())
            self.avg_slice_load_ratio.append(self.get_avg_slice_load_ratio())
            self.avg_slice_client_count.append(self.get_avg_slice_client_count())
            self.coverage_ratio.append(self.get_coverage_ratio())

            # Track base station capacities
            for bs in self.base_stations:
                total_capacity = sum(sl.capacity.level for sl in bs.slices)
                self.base_station_capacities[bs.pk].append(total_capacity)

            # Adjust base station capacity based on block ratio thresholds
            if len(self.block_count) > 10:  # Ensure enough data points
                avg_block_ratio = mean(self.block_count[-10:])
                if avg_block_ratio > self.block_ratio_upper_threshold:
                    self.increase_base_station_capacity()
                elif avg_block_ratio < self.block_ratio_lower_threshold:
                    self.decrease_base_station_capacity()

            # Prepare state for Q-learning
            state = tuple([sum(sl.capacity.level for sl in bs.slices) for bs in self.base_stations] + [self.block_count[-1]])

            # Choose action using epsilon-greedy policy
            action = self.choose_action(state)

            # Allocate resources based on the chosen action
            self.allocate_resources(action)

            # Calculate reward (negative block ratio)
            reward = -self.block_count[-1]

            # Prepare next state
            next_state = tuple([sum(sl.capacity.level for sl in bs.slices) for bs in self.base_stations] + [self.block_count[-1]])

            # Update Q-table
            self.update_q_table(state, action, reward, next_state)

            # Decay exploration rate
            if self.exploration_rate > self.exploration_min:
                self.exploration_rate *= self.exploration_decay

            self.connect_attempt.append(0)
            self.block_count.append(0)
            self.handover_count.append(0)
            yield self.env.timeout(1)

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.exploration_rate or state not in self.q_table:
            return random.randint(0, len(self.base_stations) - 1)  # Random action
        return max(self.q_table[state], key=self.q_table[state].get)  # Best action

    def update_q_table(self, state, action, reward, next_state):
        """Update Q-table using the Q-learning formula."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in range(len(self.base_stations))}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in range(len(self.base_stations))}

        best_next_action = max(self.q_table[next_state], key=self.q_table[next_state].get)
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

    def allocate_resources(self, action):
        """Allocate resources based on the chosen action."""
        for i, bs in enumerate(self.base_stations):
            if i == action:
                for sl in bs.slices:
                    additional_capacity = sl.capacity_bandwidth * self.capacity_adjustment_factor
                    sl.capacity.put(additional_capacity)
                    sl.capacity._capacity += additional_capacity
                    print(f"Q-learning allocated additional capacity to slice {sl.name} at BaseStation {bs.pk}. New capacity: {sl.capacity.capacity}")

    def increase_base_station_capacity(self):
        """Increase base station capacity."""
        for bs in self.base_stations:
            for sl in bs.slices:
                additional_capacity = sl.capacity_bandwidth * self.capacity_adjustment_factor
                sl.capacity.put(additional_capacity)
                sl.capacity._capacity += additional_capacity
                print(f"Increased capacity for slice {sl.name} at BaseStation {bs.pk}. New capacity: {sl.capacity.capacity}")

    def decrease_base_station_capacity(self):
        """Decrease base station capacity."""
        for bs in self.base_stations:
            for sl in bs.slices:
                reduction_capacity = sl.capacity_bandwidth * self.capacity_adjustment_factor
                if sl.capacity.level >= reduction_capacity:  # Ensure we don't reduce below current usage
                    sl.capacity.get(reduction_capacity)
                    sl.capacity._capacity -= reduction_capacity
                    print(f"Decreased capacity for slice {sl.name} at BaseStation {bs.pk}. New capacity: {sl.capacity.capacity}")

    def get_total_connected_users_ratio(self):
        t, cc = 0, 0
        for c in self.clients:
            if self.is_client_in_coverage(c):
                t += c.connected
                cc += 1
        # for bs in self.base_stations:
        #     for sl in bs.slices:
        #         t += sl.connected_users
        return t/cc if cc != 0 else 0

    def get_total_used_bw(self):
        t = 0
        for bs in self.base_stations:
            for sl in bs.slices:
                t += sl.capacity.capacity - sl.capacity.level
        return t

    def get_avg_slice_load_ratio(self):
        t, c = 0, 0
        for bs in self.base_stations:
            for sl in bs.slices:
                c += sl.capacity.capacity
                t += sl.capacity.capacity - sl.capacity.level
                #c += 1
                #t += (sl.capacity.capacity - sl.capacity.level) / sl.capacity.capacity
        return t/c if c !=0 else 0

    def get_avg_slice_client_count(self):
        t, c = 0, 0
        for bs in self.base_stations:
            for sl in bs.slices:
                c += 1
                t += sl.connected_users
        return t/c if c !=0 else 0
    
    def get_coverage_ratio(self):
        t, cc = 0, 0
        for c in self.clients:
            if self.is_client_in_coverage(c):
                cc += 1
                if c.base_station is not None and c.base_station.coverage.is_in_coverage(c.x, c.y):
                    t += 1
        return t/cc if cc !=0 else 0

    def incr_connect_attempt(self, client):
        if self.is_client_in_coverage(client):
            self.connect_attempt[-1] += 1

    def incr_block_count(self, client):
        if self.is_client_in_coverage(client):
            self.block_count[-1] += 1

    def incr_handover_count(self, client):
        if self.is_client_in_coverage(client):
            self.handover_count[-1] += 1

    def is_client_in_coverage(self, client):
        xs, ys = self.area
        return True if xs[0] <= client.x <= xs[1] and ys[0] <= client.y <= ys[1] else False

    def adjust_base_station_capacity(self):
        """Adjust base station capacity using Bayesian optimization."""
        def objective_function(scaling_factors):
            # Apply scaling factors to base station capacities
            for scale, bs in zip(scaling_factors, self.base_stations):
                for sl in bs.slices:
                    new_capacity = sl.capacity_bandwidth * scale
                    current_capacity = sl.capacity.capacity
                    if new_capacity > current_capacity:
                        sl.capacity.put(new_capacity - current_capacity)  # Add the increased capacity
                    elif new_capacity < current_capacity:
                        sl.capacity.get(current_capacity - new_capacity)  # Reduce the capacity
                    sl.capacity._capacity = new_capacity  # Update the internal capacity value
            # Return the average block ratio as the objective to minimize
            return mean(self.block_count[-10:]) if len(self.block_count) >= 10 else 1.0

        # Perform Bayesian optimization
        result = gp_minimize(objective_function, self.optimization_space, n_calls=10, random_state=42)
        self.optimization_results.append(result)

        # Apply the best scaling factors
        for scale, bs in zip(result.x, self.base_stations):
            for sl in bs.slices:
                new_capacity = sl.capacity_bandwidth * scale
                current_capacity = sl.capacity.capacity
                if new_capacity > current_capacity:
                    sl.capacity.put(new_capacity - current_capacity)  # Add the increased capacity
                elif new_capacity < current_capacity:
                    sl.capacity.get(current_capacity - new_capacity)  # Reduce the capacity
                sl.capacity._capacity = new_capacity  # Update the internal capacity value
                print(f"Optimized capacity for slice {sl.name} at BaseStation {bs.pk}. New capacity: {sl.capacity.capacity}")

    def allocate_resources_based_on_priority(self):
        """Allocate resources to slices based on their priority."""
        for bs in self.base_stations:
            total_available_capacity = sum(sl.capacity.level for sl in bs.slices)
            if total_available_capacity > 0:
                # Sort slices by QoS class (lower value indicates higher priority)
                sorted_slices = sorted(bs.slices, key=lambda sl: sl.qos_class)
                for sl in sorted_slices:
                    # Allocate resources proportionally based on guaranteed bandwidth
                    allocation = (sl.bandwidth_guaranteed / bs.capacity_bandwidth) * total_available_capacity
                    if sl.capacity.level < allocation:
                        additional_capacity = allocation - sl.capacity.level
                        sl.capacity.put(additional_capacity)
                        sl.capacity._capacity += additional_capacity
                        print(f"Allocated additional capacity to slice {sl.name} at BaseStation {bs.pk}. New capacity: {sl.capacity.capacity}")

    def dynamic_adjustment_loop(self):
        """Dynamically adjust thresholds and capacity adjustment factor."""
        while True:
            yield self.env.timeout(10)  # Adjust every 10 simulation seconds

            # Calculate recent average block ratio
            if len(self.block_count) > 10:
                recent_block_ratio = mean(self.block_count[-10:])
            else:
                recent_block_ratio = mean(self.block_count) if self.block_count else 0

            # Adjust thresholds based on recent block ratio
            if recent_block_ratio > self.block_ratio_upper_threshold:
                self.block_ratio_upper_threshold += 0.01  # Increase upper threshold
                self.capacity_adjustment_factor += 0.01  # Increase adjustment factor
            elif recent_block_ratio < self.block_ratio_lower_threshold:
                self.block_ratio_lower_threshold -= 0.01  # Decrease lower threshold
                self.capacity_adjustment_factor -= 0.01  # Decrease adjustment factor

            # Ensure thresholds and adjustment factor remain within valid bounds
            self.block_ratio_upper_threshold = max(0.01, min(0.1, self.block_ratio_upper_threshold))
            self.block_ratio_lower_threshold = max(0.001, min(0.05, self.block_ratio_lower_threshold))
            self.capacity_adjustment_factor = max(0.1, min(0.5, self.capacity_adjustment_factor))

            print(f"[{int(self.env.now)}] Adjusted thresholds: upper={self.block_ratio_upper_threshold:.3f}, "
                  f"lower={self.block_ratio_lower_threshold:.3f}, adjustment_factor={self.capacity_adjustment_factor:.3f}")

def optimize_thresholds(stats):
    def objective(params):
        upper_threshold, lower_threshold, adjustment_factor = params
        stats.block_ratio_upper_threshold = upper_threshold
        stats.block_ratio_lower_threshold = lower_threshold
        stats.capacity_adjustment_factor = adjustment_factor
        
        # Run a simulation and return the average block ratio
        avg_block_ratio = stats.run_simulation()  # Replace with actual simulation logic
        return avg_block_ratio

    space = [
        Real(0.01, 0.1, name='block_ratio_upper_threshold'),
        Real(0.001, 0.05, name='block_ratio_lower_threshold'),
        Real(0.1, 0.5, name='capacity_adjustment_factor')
    ]

    result = gp_minimize(objective, space, n_calls=20, random_state=42)
    return result.x  # Optimal parameters
