# NXP AIM India 2025 - Warehouse Treasure Hunt Solution

## NXP AIM competiton description
## Overview and Challenge
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

## Key Technologies and Skills
Participants gain hands-on experience in:
- **ROS 2** - Robot Operating System 2 development
- **AI/ML** - Artificial Intelligence and Machine Learning
- **Sensors** - LiDAR and camera integration
- **Navigation** - Path planning with Nav2
- **Perception** - Sensor fusion techniques
- **Control** - Real-time decision making
- **Simulation** - Gazebo environment

## Implementation Logic
### Frontier-Based Exploration and Mapping

The autonomous navigation system implements a four-phase approach to warehouse exploration and inventory management. The first phase uses frontier-based exploration to autonomously map the unknown environment. Frontiers are defined as boundaries between explored free space and unexplored unknown areas. The algorithm scans the occupancy grid to identify unknown cells that have at least one free space neighbor but no obstacle neighbors within an 8-cell radius, ensuring safe navigation paths.

The exploration strategy employs adaptive distance-based goal selection, initially searching for frontiers within 4 to 7 meters from the robot's current position. When no suitable frontiers are found within this range, the system dynamically expands the search distance by incrementally increasing the maximum range and decreasing the minimum range. The robot navigates to the closest valid frontier using Euclidean distance calculations. Once the system detects no new frontiers for consecutive cycles, it marks exploration as complete and transitions to shelf detection.

The implementation uses a two-map system: the SLAM-generated map for shelf detection and the Nav2 global costmap for frontier-based navigation. Coordinate transformations between grid indices and world coordinates enable seamless navigation goal creation.

### Shelf Detection Using PCA

After completing the exploration phase, the system identifies warehouse shelves using Principal Component Analysis. The algorithm extracts occupied cells from the SLAM map and groups them into connected components. For each component larger than the minimum cluster size, it converts pixel coordinates to metric coordinates and centers the point cloud.

Singular Value Decomposition is applied to determine the principal axes of each cluster, which represent the shelf's orientation and dimensions. By projecting the coordinates onto these principal axes, the system calculates the length and width of each structure. Validation criteria filter out non-shelf objects by checking dimension ranges and aspect ratios, ensuring only elongated rectangular structures are classified as shelves.

For each detected shelf, the system calculates two optimal viewpoints positioned 2.5 meters away: the major axis viewpoint perpendicular to the long edge for wide-angle object detection, and the minor axis viewpoint perpendicular to the short edge for close-up QR code scanning.

### Sequential Navigation

The robot visits shelves sequentially using angular matching from the world origin. An initial angle parameter identifies the first shelf, and subsequent shelves are determined by QR codes decoded at each location. At each shelf, the robot first navigates to the major axis viewpoint for object detection, spending five seconds collecting YOLO-based detections. It then moves to the minor axis viewpoint for QR code scanning during another five-second window. The QR code contains the angle to the next shelf, enabling ordered traversal. Object detections from multiple observations are merged using a maximum count strategy to handle occlusions and ensure accurate inventory counts.

