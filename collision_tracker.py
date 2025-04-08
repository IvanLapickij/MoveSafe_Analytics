# collision_tracker.py
import time

class CollisionTracker:
    def __init__(self, threshold=50):
        self.threshold = threshold
        self.current_distance = None
        self.collision_state = False
        self.collision_start_time = None
        self.current_collision_duration = 0
        self.collision_durations = []

    def update(self, distance):
        self.current_distance = distance
        if distance is not None and distance < self.threshold:
            if not self.collision_state:
                self.collision_state = True
                self.collision_start_time = time.time()
            self.current_collision_duration = time.time() - self.collision_start_time
        else:
            if self.collision_state:
                self.collision_durations.append(time.time() - self.collision_start_time)
                self.collision_state = False
                self.current_collision_duration = 0

    def reset(self):
        self.current_distance = None
        self.collision_state = False
        self.collision_start_time = None
        self.current_collision_duration = 0
        self.collision_durations = []
