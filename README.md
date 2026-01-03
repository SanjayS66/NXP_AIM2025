# 🏆 NXP AIM India 2025 - Warehouse Treasure Hunt Solution


https://github.com/user-attachments/assets/41b5d116-516e-43a9-810a-42658bbbcc90



## 🎯 NXP AIM competiton description
## 🔍 Overview and Challenge
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

## 🛠️ Key Technologies and Skills
Participants gain hands-on experience in:
- **ROS 2** - Robot Operating System 2 development
- **AI/ML** - Artificial Intelligence and Machine Learning
- **Sensors** - LiDAR and camera integration
- **Navigation** - Path planning with Nav2
- **Perception** - Sensor fusion techniques
- **Control** - Real-time decision making
- **Simulation** - Gazebo environment

## 💡 Implementation Logic
### 🗺️ Frontier-Based Exploration and Mapping

The autonomous navigation system implements a four-phase approach to warehouse exploration and inventory management. The first phase uses frontier-based exploration to map the unknown environment, where frontiers represent boundaries between explored free space and unexplored regions. The algorithm identifies frontier cells by verifying an unknown cell has at least one free space neighbor and no adjacent obstacles for safe navigation.

The exploration strategy employs adaptive distance-based goal selection, initially searching frontiers within 4 to 7 meters from the robot. When no suitable frontiers are found, the system dynamically expands the maximum search distance by 2 meters and reduces the minimum by 1 meter (0.25-meter lower limit) for comprehensive coverage. Exploration concludes after three consecutive iterations with no new frontiers detected, indicating complete mapping.

The implementation uses a two-map system: the SLAM-generated map for shelf detection and the Nav2 global costmap for frontier-based navigation, with coordinate transformations handled using the map's resolution and origin parameters for accurate pose calculations.

### 📦 Shelf Detection Using PCA

After completing exploration, the system identifies warehouse shelves using Principal Component Analysis. The algorithm extracts occupied cells from the SLAM map and groups them into connected components using scipy's label function, analyzing clusters with at least 50 pixels as potential shelf candidates.

Singular Value Decomposition determines each cluster's principal axes, representing the shelf's orientation and dimensions. By projecting coordinates onto these axes and calculating peak-to-peak spread, the system measures width and height. Validation filters ensure detected objects have shelf-like characteristics: long dimension between 0.8-15 meters, short dimension between 0.2-3 meters, and aspect ratio between 1.5-15 for elongated rectangular structures.

For each detected shelf, the system calculates two optimal viewpoints positioned 2.5 meters away: the major axis viewpoint perpendicular to the long edge for wide-angle object detection, and the minor axis viewpoint perpendicular to the short edge for QR code scanning, with precise orientation angles ensuring the robot faces the shelf for optimal sensor coverage.

### 🚀 Sequential Navigation

The robot visits shelves sequentially using angular matching from the world origin. An initial angle parameter identifies the first shelf by calculating the angle from the origin (0,0) to each shelf's center using `atan2(cy, cx)` and selecting the shelf with minimal angular error. This ensures deterministic starting behavior regardless of the robot's current position.

Once the first shelf is identified, the robot navigates to its major axis viewpoint for object detection. The system employs a two-phase inspection approach at each shelf:

**Phase 1 - Object Detection (Major Axis):**  
The robot positions itself perpendicular to the shelf's long edge and activates YOLO-based object detection for 5 seconds.  Detected objects are merged using a "maximum count" rule across multiple detections to handle occlusions and varying viewing angles. Only objects from the allowed list (banana, zebra, teddy bear, car, potted plant, cup, clock, horse) are retained. 

**Phase 2 - QR Code Scanning (Minor Axis):**  
After object detection completes, the robot moves to the minor axis viewpoint perpendicular to the shelf's short edge. The camera callback activates QR scanning using the pyzbar library for 5 seconds. When a QR code is successfully decoded, the system parses the angle value (format: `prefix_angle`) and updates the `initial_angle` parameter for subsequent shelf matching.

Upon completing both phases, the current shelf is marked as visited, and its data (object names, counts, and QR code) is published to the `/shelf_data` topic. The `navigate_to_next_shelf()` function then uses the QR-decoded angle to identify the next unvisited shelf with minimal angular error. This process repeats until either no QR data is available or all shelves have been visited, at which point the system logs "Workflow complete!" and terminates shelf navigation.

**Error Handling:**  
Angular error is calculated using the formula `abs((angle_deg - initial_angle + 180) % 360 - 180)` to handle wraparound at 0°/360° correctly. If no matching shelf is found or the QR code contains "Empty", the navigation sequence concludes gracefully.  The system also includes timer management to prevent blocking operations, destroying previous timers before creating new ones for each navigation phase.

## 📊 Results and Performance

Our team successfully cleared the challenging simulations round with excellent performance, demonstrating strong technical capabilities in autonomous navigation and ROS2 implementation. Based on our results, we were selected among the top teams to compete in the Regional Finale held at Bangalore, where we had the opportunity to showcase our solution on real hardware and compete against other talented teams from across the region.

## 🚀 How to Run

```bash
# Clone the repository inside the ROS2 workspace's src directory by running following command
git clone https://github.com/SanjayS66/NXP_AIM2025

# Build the workspace
colcon build

# Source the workspace
source install/setup. bash

# Launch with parameters
ros2 launch b3rb_ros_aim_india warehouse_navigation.launch.py shelf_count:=4 initial_angle:=45. 0
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
