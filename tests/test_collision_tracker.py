import time
from collision_tracker import CollisionTracker

def test_collision_logic():
    tracker = CollisionTracker(threshold=50)
    
    tracker.update(45)  # Under threshold — collision starts
    time.sleep(0.1)
    tracker.update(45)  # Still colliding
    time.sleep(0.1)
    tracker.update(100)  # Ends collision
    
    assert tracker.collision_durations, "Should record at least one collision"
    assert tracker.current_collision_duration == 0
    assert tracker.collision_state is False
