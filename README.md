# NXP AIM India 2025 - Warehouse Treasure Hunt Solution

### NXP AIM competiton description
### Overview and Challenge
The NXP AIM India 2025 (Artificial Intelligence in Mobility) competition is a nationwide challenge by NXP Semiconductors that invites engineering students across India to: 
- Design and test intelligent autonomous mobility solutions
- Use simulation and modern robotics technologies
- Address real-world mobility challenges

**The Core Challenge:** Develop an autonomous mobile robotics solution that can:  
- **Navigate efficiently** in a simulated warehouse environment
- **Perform inventory management tasks** autonomously
- **Integrate sensor data** from cameras and LiDAR for robust perception
- **Avoid obstacles** using intelligent path planning
- **Complete missions quickly** with minimal penalties
- **Demonstrate real-time decision-making** and control in complex scenarios

### Key Technologies and Skills
Participants gain hands-on experience in:
- **ROS 2** - Robot Operating System 2 development
- **AI/ML** - Artificial Intelligence and Machine Learning
- **Sensors** - LiDAR and camera integration
- **Navigation** - Path planning with Nav2
- **Perception** - Sensor fusion techniques
- **Control** - Real-time decision making
- **Simulation** - Gazebo environment

---

## Implementation: Frontier-Based Planning and Mapping

This section documents the autonomous navigation and mapping implementation used in `b3rb_ros_warehouse.py` for the warehouse treasure hunt challenge.

### Architecture Overview

The solution implements a multi-phase autonomous exploration and inventory management system:

1. **Phase 1: Frontier-Based Exploration** - Autonomous map building using frontier detection
2. **Phase 2: Shelf Detection** - Identifying warehouse shelves using PCA (Principal Component Analysis)
3. **Phase 3: Sequential Navigation** - Visiting shelves in order guided by QR codes
4. **Phase 4: Object Detection & QR Scanning** - Inventory management at each shelf

### Core Components

#### 1. Frontier-Based Space Exploration

**Purpose**: Autonomously explore and map the unknown warehouse environment.

**Algorithm** (`get_frontiers_for_space_exploration`):
```
For each cell in the costmap:
  1. If cell is UNKNOWN (-1):
     2. Check if any CARDINAL neighbor is FREE (0)
     3. Check if any COMPLETE neighbor is OBSTACLE (>0)
     4. If has FREE neighbor AND no OBSTACLE neighbors:
        - Mark as frontier cell
```

**Key Features**:
- **Frontier Definition**: Boundary between explored free space and unexplored unknown space
- **Safety Check**: Excludes frontiers near obstacles to ensure safe navigation
- **Cardinal Neighbors**: Up, down, left, right (4-connectivity for free space check)
- **Complete Neighbors**: All 8 surrounding cells (for obstacle avoidance)

**Implementation Details**:
```python
# From b3rb_ros_warehouse.py lines 580-624
def get_frontiers_for_space_exploration(self, map_array):
    frontiers = []
    for y in range(1, map_array.shape[0] - 1):
        for x in range(1, map_array.shape[1] - 1):
            if map_array[y, x] == -1:  # Unknown space
                # Check 8 neighbors for obstacles
                neighbors_complete = [(y±1, x±1), (y±1, x), (y, x±1)]
                # Check 4 cardinal neighbors for free space
                neighbors_cardinal = [(y±1, x), (y, x±1)]
                
                if has_free_neighbor AND no_obstacle_neighbors:
                    frontiers.append((ny, nx))
    return frontiers
```

#### 2. Adaptive Exploration Strategy

**Distance-Based Goal Selection** (`global_map_callback`):
- Searches for closest frontier within a distance range
- Initial range: 4.0m to 7.0m from robot position
- Uses Euclidean distance for frontier ranking
- Dynamically adjusts search range if no frontiers found:
  - Increases max distance by 2.0m
  - Decreases min distance by 1.0m (lower bound: 0.25m)

**Exploration Completion**:
- Monitors `full_map_explored_count` when no frontiers detected
- After 2+ consecutive cycles with no frontiers → triggers Phase 2
- Prevents infinite exploration loops

```python
# Lines 286-357
if frontiers:
    # Find closest frontier in distance range
    closest_frontier = select_frontier_in_range(
        min_dist=self.min_step_dist_world_meters,  # 4.0m initially
        max_dist=self.max_step_dist_world_meters   # 7.0m initially
    )
    if closest_frontier:
        send_goal_to_frontier()
    else:
        # Expand search range
        self.max_step_dist_world_meters += 2.0
        self.min_step_dist_world_meters = max(0.25, 
                                              self.min_step_dist_world_meters - 1.0)
else:
    self.full_map_explored_count += 1
    if self.full_map_explored_count > 2:
        # Trigger shelf detection phase
        self.exploration_done = True
        self.shelves_final = self.shelf_detection(self.simple_map_curr)
```

#### 3. SLAM and Map Integration

**Two-Map System**:

| Map Type | Topic | Purpose | Occupancy Values |
|----------|-------|---------|------------------|
| **Simple Map** | `/map` | SLAM output, used for shelf detection | 0=free, 100=occupied, -1=unknown |
| **Global Costmap** | `/global_costmap/costmap` | Nav2 planning costmap with inflation | Continuous 0-100 with inflated obstacles |

**Coordinate Transformations**:
```python
# Lines 1106-1126
def get_world_coord_from_map_coord(self, map_x, map_y, map_info):
    """Convert grid coordinates to world frame (meters)"""
    resolution = map_info.resolution  # meters per cell
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    
    world_x = (map_x + 0.5) * resolution + origin_x
    world_y = (map_y + 0.5) * resolution + origin_y
    return (world_x, world_y)

def get_map_coord_from_world_coord(self, world_x, world_y, map_info):
    """Convert world coordinates to grid indices"""
    map_x = int((world_x - origin_x) / resolution)
    map_y = int((world_y - origin_y) / resolution)
    return (map_x, map_y)
```

#### 4. Shelf Detection Using PCA

**Purpose**: Identify elongated rectangular structures (warehouse shelves) from the occupancy grid.

**Algorithm** (`shelf_detection`, lines 359-442):
```
1. Extract occupied cells (value = 100) from SLAM map
2. Find connected components using scipy.ndimage.label()
3. For each component with size > MIN_CLUSTER_SIZE (50 pixels):
   
   a. Convert pixel coordinates to metric coordinates
   b. Center the coordinate cloud
   c. Apply Singular Value Decomposition (SVD):
      - U, Σ, V^T = SVD(centered_coords)
      - Principal axes = columns of V^T
   
   d. Project coordinates onto principal axes
   e. Calculate dimensions:
      - width = range along 1st principal axis
      - height = range along 2nd principal axis
   
   f. Validate shelf characteristics:
      - Long dimension: 0.8m to 15.0m
      - Short dimension: 0.2m to 3.0m
      - Aspect ratio: 1.5 to 15.0 (elongated)
   
   g. Calculate orientation from principal axis:
      - orientation = atan2(axis_y, axis_x)
   
   h. Store shelf data: {center, dimensions, orientation}
```

**PCA Benefits**:
- Automatically finds shelf orientation regardless of angle
- Robust to irregular occupancy grid edges
- Provides accurate center point and dimensions
- Filters out non-shelf objects (circular obstacles, walls)

**Validation Criteria** (`_is_valid_pca_shelf`, lines 444-473):
```python
MIN_LONG_DIM = 0.8      # Minimum shelf length
MAX_LONG_DIM = 15.0     # Maximum shelf length
MIN_SHORT_DIM = 0.2     # Minimum shelf width
MAX_SHORT_DIM = 3.0     # Maximum shelf width
MIN_ASPECT = 1.5        # Must be elongated
MAX_ASPECT = 15.0       # Not too thin (noise)
```

#### 5. Viewpoint Calculation Strategy

**Purpose**: Calculate optimal robot positions for object detection and QR scanning at each shelf.

**Two Viewpoints Per Shelf** (`calculate_shelf_viewpoints`, lines 476-506):

1. **Major Axis Viewpoint** (Object Detection):
   - Position: Perpendicular to shelf's **long edge**
   - Distance: 2.5m from shelf center
   - Orientation: Facing the long side of shelf
   - Purpose: Wide field of view for object detection via YOLO

2. **Minor Axis Viewpoint** (QR Scanning):
   - Position: Perpendicular to shelf's **short edge**
   - Distance: 2.5m from shelf center
   - Orientation: Facing the short end of shelf
   - Purpose: Close-up view for QR code reading

**Calculation**:
```python
# Extract shelf orientation from PCA
orientation = shelf['orientation']  # radians

# Long axis direction (shelf's length)
long_axis_x = cos(orientation)
long_axis_y = sin(orientation)

# Short axis direction (perpendicular to length)
short_axis_x = -long_axis_y
short_axis_y = long_axis_x

# Major viewpoint (for object detection)
major_x = center_x + 2.5 * short_axis_x
major_y = center_y + 2.5 * short_axis_y
major_angle = atan2(-short_axis_y, -short_axis_x)

# Minor viewpoint (for QR scanning)
minor_x = center_x + 2.5 * long_axis_x
minor_y = center_y + 2.5 * long_axis_y
minor_angle = atan2(-long_axis_y, -long_axis_x)
```

#### 6. Sequential Navigation with QR Guidance

**Navigation Workflow**:

```
START: Robot at origin (0, 0)
  ↓
STEP 1: Navigate to first shelf using initial_angle parameter
  - Calculate angle from origin to each shelf center
  - Select shelf with smallest angular error
  - Go to major_axis viewpoint
  ↓
STEP 2: Object Detection at Major Axis (5 seconds)
  - Receive YOLO detections via /shelf_objects topic
  - Merge detection results (max count per object)
  ↓
STEP 3: Navigate to Minor Axis
  - Move to minor_axis viewpoint
  ↓
STEP 4: QR Code Scanning (5 seconds)
  - Decode QR code from camera image
  - Extract next shelf angle: "shelf_X_<angle>"
  - Update initial_angle = extracted angle
  ↓
STEP 5: Publish Shelf Data
  - Combine objects + QR code
  - Publish to /shelf_data topic
  - Mark shelf as visited
  ↓
STEP 6: Find Next Shelf
  - Calculate angle from origin to each unvisited shelf
  - Select shelf matching QR-decoded angle
  - If match found → Go to STEP 2
  - If no shelves left → END
```

**First Shelf Selection** (`navigate_to_first_shelf`, lines 533-578):
```python
# Use launch parameter initial_angle (e.g., 45°, 135°, 225°, 315°)
for shelf in shelves_final:
    angle_to_shelf = atan2(shelf_y - 0.0, shelf_x - 0.0)  # From origin
    error = abs(angle_to_shelf - initial_angle)
    
    if error < min_error:
        best_shelf = shelf
```

**Next Shelf Selection** (`navigate_to_next_shelf`, lines 723-774):
```python
# Use angle extracted from QR code
qr_angle = extract_angle_from_qr(self.qr_data)  # e.g., "shelf_2_135" → 135°

for shelf in shelves_final:
    if not shelf['visited']:
        angle_to_shelf = atan2(shelf_y - 0.0, shelf_x - 0.0)
        error = abs(angle_to_shelf - qr_angle)
        
        if error < min_error:
            next_shelf = shelf
```

#### 7. Object Detection Merging

**Purpose**: Combine multiple YOLO detections at major axis viewpoint to get accurate inventory counts.

**Merge Strategy** (`merge_object_detections`, lines 778-817):
```python
# Allowed objects in competition
ALLOWED_OBJECTS = {
    "banana", "zebra", "teddy bear", "car",
    "potted plant", "cup", "clock", "horse"
}

# Merge rule: MAX count from multiple detections
for object_name in detected_objects:
    if object_name in ALLOWED_OBJECTS:
        prev_count = previous_detections[object_name]
        curr_count = current_detection[object_name]
        merged_count = max(prev_count, curr_count)
```

**Why MAX and not SUM?**
- Handles detection inconsistencies (objects temporarily occluded)
- Prevents double-counting same objects from multiple frames
- More robust to YOLO false negatives

#### 8. Navigation Goal Management

**Nav2 Integration** (lines 67-1093):
- Uses NavigateToPose action client
- Asynchronous goal handling with callbacks
- Recovery mechanism: Cancel goal after 20 recovery attempts
- Goal tolerance: 0.5m xy_goal_tolerance

**Goal Creation Pipeline**:
```
World Coordinates (x, y, yaw)
    ↓
create_goal_from_world_coord()
    ↓
PoseStamped Message
    ↓
NavigateToPose.Goal
    ↓
Nav2 Action Server
```

### Data Flow Diagram

```
┌─────────────────┐
│  SLAM (/map)    │──→ Shelf Detection (PCA)
└─────────────────┘        ↓
                     Shelf Locations + Orientations
                            ↓
┌─────────────────┐    Calculate Viewpoints
│ Global Costmap  │        ↓
│ /global_costmap │←─ Frontier Detection → Navigation Goals
└─────────────────┘        ↓
                      Explore Until Complete
                            ↓
                      Start Shelf Sequence
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
   Major Axis                              Minor Axis
   (Object Detection)                      (QR Scanning)
        ↓                                       ↓
   /shelf_objects ←─ YOLO ─→ Merge         Decode QR
        ↓                                       ↓
   Object List                           Next Shelf Angle
        └───────────────────┬───────────────────┘
                            ↓
                   Publish /shelf_data
                            ↓
                   Navigate to Next Shelf
```

### Key Parameters

| Parameter | Default Value | Purpose |
|-----------|--------------|---------|
| `max_step_dist_world_meters` | 7.0 | Maximum distance to frontier |
| `min_step_dist_world_meters` | 4.0 | Minimum distance to frontier |
| `xy_goal_tolerance` | 0.5 | Goal completion radius (meters) |
| `recovery_threshold` | 20 | Max recovery attempts before cancel |
| `MIN_CLUSTER_SIZE` | 50 | Min pixels for shelf candidate |
| `viewpoint_distance` | 2.5 | Distance from shelf to viewpoint (meters) |
| `shelf_count` | 1 | Number of shelves (launch param) |
| `initial_angle` | 0.0 | First shelf angle in degrees (launch param) |

### Subscribed Topics

| Topic | Message Type | Purpose |
|-------|--------------|---------|
| `/pose` | PoseWithCovarianceStamped | Robot localization |
| `/map` | OccupancyGrid | SLAM map for shelf detection |
| `/global_costmap/costmap` | OccupancyGrid | Navigation costmap for frontier detection |
| `/shelf_objects` | WarehouseShelf | YOLO object detection results |
| `/camera/image_raw/compressed` | CompressedImage | Camera feed for QR scanning |
| `/cerebri/out/status` | Status | Robot arming status |
| `/behavior_tree_log` | BehaviorTreeLog | Nav2 behavior tree events |

### Published Topics

| Topic | Message Type | Purpose |
|-------|--------------|---------|
| `/shelf_data` | WarehouseShelf | Final shelf inventory (objects + QR code) |
| `/cerebri/in/joy` | Joy | Manual control commands (arming) |
| `/debug_images/qr_code` | CompressedImage | Debug QR detection (optional) |

### Algorithm Complexity

- **Frontier Detection**: O(W × H) where W, H = map dimensions
- **Frontier Selection**: O(F) where F = number of frontiers (typically < 100)
- **Shelf Detection**: O(N × K) where N = occupied cells, K = components (typically < 10)
- **PCA per Shelf**: O(P²) where P = points in component (~50-500)

### Edge Cases Handled

1. **No Frontiers Found**: Incrementally expand search distance range
2. **Multiple Shelf Detection Attempts**: Validates dimensions and aspect ratios
3. **Goal Rejection**: Logs warning and marks goal as completed
4. **Excessive Recoveries**: Cancels stuck navigation goals
5. **QR Parsing Failures**: Gracefully handles malformed QR data
6. **Missing Objects**: Merges detections using MAX to handle occlusions

### Performance Considerations

- Frontier detection runs only when `goal_completed = True` (not during active navigation)
- Shelf detection runs once after exploration phase
- Object detection merging happens in real-time during shelf navigation
- QR scanning is time-boxed to 5-second windows

---

## Conclusion

This implementation demonstrates a complete autonomous warehouse navigation system combining:
- Classical robotics algorithms (frontier exploration)
- Modern ML techniques (YOLO object detection)
- Computer vision (QR code scanning, PCA for shelf detection)
- ROS 2 navigation stack integration (Nav2)

The frontier-based approach ensures complete map coverage while the PCA-based shelf detection provides robust identification of storage structures regardless of orientation. Sequential navigation guided by QR codes enables efficient inventory management across the warehouse.


