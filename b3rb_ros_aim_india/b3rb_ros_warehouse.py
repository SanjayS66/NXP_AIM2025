# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

from pyzbar.pyzbar import decode

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True

#GUI part : Tkinter 
class WindowProgressTable:
    def __init__(self, root, shelf_count):
        self.root = root
        self.root.title("Shelf Objects & QR Link")
        self.root.attributes("-topmost", True)

        self.row_count = 2
        self.col_count = shelf_count

        self.boxes = []
        for row in range(self.row_count):
            row_boxes = []
            for col in range(self.col_count):
                box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
                          relief="solid", font=("Helvetica", 14))
                box.insert(tk.END, "NULL")
                box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
                row_boxes.append(box)
            self.boxes.append(row_boxes)

        # Make the grid layout responsive.
        for row in range(self.row_count):
            self.root.grid_rowconfigure(row, weight=1)
        for col in range(self.col_count):
            self.root.grid_columnconfigure(col, weight=1)

    def change_box_color(self, row, col, color):
        self.boxes[row][col].config(bg=color)

    def change_box_text(self, row, col, text):
        self.boxes[row][col].delete(1.0, tk.END)
        self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
    global box_app
    root = tk.Tk()
    box_app = WindowProgressTable(root, shelf_count)
    root.mainloop()

#main ros2 node
class WarehouseExplore(Node):
    """ Initializes warehouse explorer node with the required publishers and subscriptions.

        Returns:
            None
    """
    def __init__(self):
        super().__init__('warehouse_explore')

        #action client to send goal pose to nav2 server
        self.action_client = ActionClient(
            self,
            NavigateToPose,				  
            '/navigate_to_pose')

        #current robot position	
        self.subscription_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.pose_callback,		      			
            QOS_PROFILE_DEFAULT)

        #costmap for path planning
        self.subscription_global_map = self.create_subscription(
            OccupancyGrid,
            '/global_costmap/costmap', 
            self.global_map_callback,
            QOS_PROFILE_DEFAULT)

        #SLAM output map
        self.subscription_simple_map = self.create_subscription(
            OccupancyGrid,
            '/map',						
            self.simple_map_callback,
            QOS_PROFILE_DEFAULT)

        #robot status
        self.subscription_status = self.create_subscription(
            Status,
            '/cerebri/out/status',		
            self.cerebri_status_callback,
            QOS_PROFILE_DEFAULT)

        self.subscription_behavior = self.create_subscription(
            BehaviorTreeLog,
            '/behavior_tree_log',
            self.behavior_tree_log_callback,
            QOS_PROFILE_DEFAULT)

        #to get object detection result from YOLO
        self.subscription_shelf_objects = self.create_subscription(
            WarehouseShelf,
            '/shelf_objects',
            self.shelf_objects_callback,
            QOS_PROFILE_DEFAULT)

        # Subscription for camera images for QR 
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            QOS_PROFILE_DEFAULT)

        #send manual control commands to bot 
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher for output image (for debug purposes).
        self.publisher_qr_decode = self.create_publisher(
            CompressedImage,
            "/debug_images/qr_code",
            QOS_PROFILE_DEFAULT)

        #publish shelf data
        self.publisher_shelf_data = self.create_publisher(
            WarehouseShelf,
            "/shelf_data",
            QOS_PROFILE_DEFAULT)


        self.declare_parameter('shelf_count', 1)         #(can be set when launching)
        self.declare_parameter('initial_angle', 0.0)     #(can be set when launching)


        #Retrieves the values of the declared parameters from ROS2 parameter server and stores them in local variables.
        # (because get_parameter is relative slow and hence stored in a local variable)
        self.shelf_count = \
            self.get_parameter('shelf_count').get_parameter_value().integer_value
        self.initial_angle = \
            self.get_parameter('initial_angle').get_parameter_value().double_value

        # --- Robot State ---
        self.armed = False   #boolean flag to track whether robot is activated/armed
        self.logger = self.get_logger()  #log data for debugging

        # --- Robot Pose ---
        self.pose_curr = PoseWithCovarianceStamped()  #pose message object having position,orientation and uncertainity
        self.buggy_pose_x = 0.0    #robots current x
        self.buggy_pose_y = 0.0    #robots current y
        self.buggy_center = (0.0, 0.0)   #robot current position as coordinate pair
        self.world_center = (0.0, 0.0)   #map center as coordinate pair

        # --- Map Data ---
        self.simple_map_curr = None   #to store the slam map (Type: nav_msgs/OccupancyGrid)
        self.global_map_curr = None   #to store the costmap (Type: nav_msgs/OccupancyGrid)

        # --- Goal Management ---
        self.xy_goal_tolerance = 0.5  #range within target pose bot is considered to reached target
        self.goal_completed = True   # No goal is currently in-progress.
        self.goal_handle_curr = None  #reference to current active goal 
        self.cancelling_goal = False  #tracks whether goal cancellation in progress or not
        self.recovery_threshold = 20  #number of recovery steps before giving up

        # --- Goal Creation ---
        self._frame_id = "map"
        
        #frontier exploration : bot moves autonomously from free space to frontier(i.e. boundary between explored and unexplored spaces ) 
        # --- Exploration Parameters ---
        self.max_step_dist_world_meters = 7.0
        self.min_step_dist_world_meters = 4.0
        self.full_map_explored_count = 0

        # --- QR Code Data ---
        self.qr_data = "Empty"
        if PROGRESS_TABLE_GUI:
            self.table_row_count = 0
            self.table_col_count = 0

        # --- Shelf Data ---
        self.shelf_objects_curr = WarehouseShelf()
        self.full_map_explored_count = 0
        self.exploration_done = False

        #shelf detection using PCA variables
        self.MIN_CLUSTER_SIZE = 50 

        #timer management variables
        self.object_timer = None
        self.qr_timer = None
        #navigation variables 
        self.shelf_navigation_active = False
        self.at_major_axis= False
        self.current_shelf = None
        self.shelves_final = []  # Store detected shelves
        self.object_names = []   # For merged object detection
        self.object_count = []   # For merged object counts
        self.current_shelf_index = 0
        self.qr_scanning = False
        self.qr_data = "Empty"

    #extract the current x and y from the pose 
    def pose_callback(self, message):#✅
        """Callback function to handle pose updates.

        Args:
            message: ROS2 message containing the current pose of the rover.

        Returns:
            None
        """
        self.pose_curr = message
        self.buggy_pose_x = message.pose.pose.position.x
        self.buggy_pose_y = message.pose.pose.position.y
        self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)
    
    #process the raw slam OccupancyGrid and calculate the world center
    def simple_map_callback(self, message):#✅
        """Callback function to handle simple map updates.

        Args:
            message: ROS2 message containing the simple map data.

        Returns:
            None
        """
        self.simple_map_curr = message
        map_info = self.simple_map_curr.info  #metdata of map i.e. width,height,res etc.
        self.world_center = self.get_world_coord_from_map_coord(
            map_info.width / 2, map_info.height / 2, map_info
        )

    def global_map_callback(self, message):
        """Callback function to handle global map updates.

        Args:
            message: ROS2 message containing the global map data.

        Returns:
            None
        """
        self.global_map_curr = message

        # if self.exploration_done:
        #     self.logger.debug("Exploration completed, skipping frontier detection")
        #     return

        if not self.goal_completed:
            return

        if self.exploration_done:
            return

        height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
        map_array = np.array(self.global_map_curr.data).reshape((height, width))

        frontiers = self.get_frontiers_for_space_exploration(map_array)

        #debug print
        self.logger.info(f"Detected {len(frontiers)} frontiers")

        map_info = self.global_map_curr.info
        if frontiers:
            closest_frontier = None
            min_distance_curr = float('inf')

            for fy, fx in frontiers:
                fx_world, fy_world = self.get_world_coord_from_map_coord(fx, fy,
                                             map_info)
                distance = euclidean((fx_world, fy_world), self.buggy_center)
                if (distance < min_distance_curr and
                    distance <= self.max_step_dist_world_meters and
                    distance >= self.min_step_dist_world_meters):
                    min_distance_curr = distance
                    closest_frontier = (fy, fx)

            if closest_frontier:
                fy, fx = closest_frontier
                goal = self.create_goal_from_map_coord(fx, fy, map_info)
                print("Sending goal for space exploration.")
                self.send_goal_from_world_pose(goal)
                return
            else:
                self.max_step_dist_world_meters += 2.0
                new_min_step_dist = self.min_step_dist_world_meters - 1.0
                self.min_step_dist_world_meters = max(0.25, new_min_step_dist)

            self.full_map_explored_count = 0
        else:
            self.full_map_explored_count += 1
            print(f"Nothing found in frontiers; count = {self.full_map_explored_count}")

#new part
        if self.full_map_explored_count > 2 and not self.exploration_done :
            
            self.exploration_done = True 
            self.logger.info("Map fully explored and begiing shelf detection!!!")
            self.shelves_final = self.shelf_detection(self.simple_map_curr)
            if self.shelves_final:
                self.logger.info("Starting Navigation to the detected shelves...")
                self.logger.info("Calculating the viewpoints for each shelves...")
                self.calculate_shelf_viewpoints()
                self.debug_shelf_angles()
                self.navigate_to_first_shelf()

    def shelf_detection(self,simple_map_curr):
        slam_map = simple_map_curr

        if not slam_map:
            self.logger.error("No map data available for shelf detection")
            return []
        
        #extract map info
        map_info = slam_map.info
        conversion_info = self._get_map_conversion_info(map_info)
        if not conversion_info:
            self.logger.error("Failed to get map conversion info")
            return []
        resolution, origin_x, origin_y = conversion_info
        
        #convert occupancy grid to numpy array
        height, width = map_info.height, map_info.width
        grid = np.array(slam_map.data, dtype = np.int8).reshape((height, width))

        #getting occupied cells 
        occupied = (grid == 100)
        labels, num_components = label(occupied)
        self.logger.info(f"Found {num_components} connected components in occupancy grid")
        detected_shelves = []
    
        for component_id in range(1, num_components + 1):
            # Extract pixel coordinates for this component
            component_pixels = np.argwhere(labels == component_id)
            
            if component_pixels.shape[0] < self.MIN_CLUSTER_SIZE:
                continue
            
            # Convert pixel coordinates to metric coordinates
            coords_metric = []
            for pixel in component_pixels:
                map_y, map_x = pixel[0], pixel[1]  # argwhere returns (row, col) = (y, x)
                world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)
                coords_metric.append([world_x, world_y])
            
            coords_metric = np.array(coords_metric)
            
            # Center the coordinates
            center_metric = coords_metric.mean(axis=0)
            centered_coords = coords_metric - center_metric
            
            # Apply PCA using SVD
            try:
                _, _, vt = np.linalg.svd(centered_coords, full_matrices=False)
                principal_axes = vt.T  # Columns are principal directions
                
                # Project coordinates onto principal axes
                projected = centered_coords @ principal_axes
                
                # Calculate dimensions along principal axes
                width_m = projected[:, 0].ptp()  # Peak-to-peak along first axis
                height_m = projected[:, 1].ptp()  # Peak-to-peak along second axis
                
                # Validate shelf characteristics
                aspect_ratio = max(width_m, height_m) / min(width_m, height_m)
                if not self._is_valid_pca_shelf(width_m, height_m, aspect_ratio):
                    continue
                
                # Calculate orientation from principal axis
                orientation_rad = math.atan2(principal_axes[1, 0], principal_axes[0, 0])
                
                # Store shelf data
                shelf_data = {
                    'center_world': (center_metric[0], center_metric[1]),
                    'dimensions_m': (width_m, height_m),
                    'orientation': orientation_rad,
                    'cluster_size': component_pixels.shape[0],
                    'aspect_ratio': aspect_ratio
                }
                
                detected_shelves.append(shelf_data)
                
                self.logger.info(f"Detected shelf: center=({center_metric[0]:.2f}, {center_metric[1]:.2f}) with dims=({width_m:.2f}x{height_m:.2f}m), orientation={math.degrees(orientation_rad):.1f}°")
                
            except np.linalg.LinAlgError:
                self.logger.warn(f"SVD failed for component {component_id}, skipping")
                continue
        
        self.logger.info(f"Successfully detected {len(detected_shelves)} shelves using PCA")
        return detected_shelves

    def _is_valid_pca_shelf(self, width_m, height_m, aspect_ratio):
        """
        Validate if detected object has shelf-like characteristics.
    
        Args:
            width_m: Width in meters
            height_m: Height in meters  
            aspect_ratio: Ratio of longer dimension to shorter dimension
    
        Returns:
            bool: True if object passes shelf validation criteria
        """
        # Define shelf constraints for warehouse environment
        MIN_LONG_DIM = 0.8      # Minimum long dimension (meters)
        MAX_LONG_DIM = 15.0     # Maximum long dimension (meters)
        MIN_SHORT_DIM = 0.2     # Minimum short dimension (meters)
        MAX_SHORT_DIM = 3.0     # Maximum short dimension (meters)
        MIN_ASPECT = 1.5        # Minimum aspect ratio (elongated)
        MAX_ASPECT = 15.0      # Maximum aspect ratio (not too thin)
        
        long_dim = max(width_m, height_m)
        short_dim = min(width_m, height_m)
        
        # Check all validation criteria
        valid_dimensions = (MIN_LONG_DIM <= long_dim <= MAX_LONG_DIM and
                        MIN_SHORT_DIM <= short_dim <= MAX_SHORT_DIM)
        
        valid_aspect = MIN_ASPECT <= aspect_ratio <= MAX_ASPECT
        self.logger.info("check for valid shelves complete")
        return valid_dimensions and valid_aspect
    
    
    def calculate_shelf_viewpoints(self):
        """Calculate optimal viewpoints for each detected shelf"""
        for shelf in self.shelves_final:  # Use shelves_final instead of self.detected_shelves
            center_x, center_y = shelf['center_world']
            orientation = shelf['orientation']
            
            # Calculate perpendicular directions
            long_axis_x = math.cos(orientation)
            long_axis_y = math.sin(orientation)
            short_axis_x = -long_axis_y
            short_axis_y = long_axis_x
            
            # Calculate viewpoint positions (2.5m away from shelf center)
            viewpoint_distance = 2.5
            
            # Long viewpoint (perpendicular to long edge, for object detection)
            major_x = center_x + viewpoint_distance * short_axis_x
            major_y = center_y + viewpoint_distance * short_axis_y
            major_angle = math.atan2(-short_axis_y, -short_axis_x)
            
            # Short viewpoint (perpendicular to short edge, for QR scanning)  
            minor_x = center_x + viewpoint_distance * long_axis_x
            minor_y = center_y + viewpoint_distance * long_axis_y
            minor_angle = math.atan2(-long_axis_y, -long_axis_x)
            
            # Store viewpoints in shelf data
            shelf['major_axis'] = (major_x , major_y , major_angle )
            shelf['minor_axis'] = (minor_x, minor_y, minor_angle)
            shelf['visited'] = False
            
            self.logger.info(f"Calculated viewpoints for shelf at ({center_x:.2f}, {center_y:.2f})")
    
    def debug_shelf_angles(self):
        """Debug function to print all shelf angles from origin"""
        self.logger.info("SHELF ANGLE ANALYSIS:")
        self.logger.info(f"Target initial_angle: {self.initial_angle}°")
        self.logger.info(f"Robot at: ({self.buggy_pose_x:.2f}, {self.buggy_pose_y:.2f})")
        
        for i, shelf in enumerate(self.shelves_final):
            cx, cy = shelf['center_world']
            
            # Angle from origin to shelf
            angle_from_origin = math.atan2(cy, cx)
            angle_from_origin_deg = (math.degrees(angle_from_origin) + 360) % 360
            
            # Angle from robot to shelf
            angle_from_robot = math.atan2(cy - self.buggy_pose_y, cx - self.buggy_pose_x)
            angle_from_robot_deg = (math.degrees(angle_from_robot) + 360) % 360
            
            # Errors
            error_from_origin = abs((angle_from_origin_deg - self.initial_angle + 180) % 360 - 180)
            error_from_robot = abs((angle_from_robot_deg - self.initial_angle + 180) % 360 - 180)
            
            self.logger.info(f"  Shelf {i+1} at ({cx:.2f}, {cy:.2f}):")
            self.logger.info(f"    From origin: {angle_from_origin_deg:.1f}° (error: {error_from_origin:.1f}°)")
            self.logger.info(f"    From robot:  {angle_from_robot_deg:.1f}° (error: {error_from_robot:.1f}°)")

    def navigate_to_first_shelf(self):
        """Navigate to first shelf using initial_angle calculated from world origin (0,0)"""
        best_shelf = None
        best_index = -1
        min_angle_error = float('inf')
        
        self.logger.info(f"Looking for first shelf using initial angle: {self.initial_angle}°")
        self.logger.info(f"Robot current position: ({self.buggy_pose_x:.2f}, {self.buggy_pose_y:.2f})")
        
        for i, shelf in enumerate(self.shelves_final):
            if shelf.get('visited', False):
                continue
                
            cx, cy = shelf['center_world']
            
            # Calculate angle from world origin (0,0) to shelf center
            angle_to_shelf = math.atan2(cy - 0.0, cx - 0.0)  # From origin to shelf
            angle_deg = (math.degrees(angle_to_shelf) + 360) % 360
            
            # Match against initial_angle
            error = abs((angle_deg - self.initial_angle + 180) % 360 - 180)
            
            self.logger.info(f"  Shelf {i+1}: center=({cx:.2f}, {cy:.2f})")
            self.logger.info(f"    Angle from origin: {angle_deg:.1f}°, error: {error:.1f}°")
            
            if error < min_angle_error:
                min_angle_error = error
                best_shelf = shelf
                best_index = i
        
        if best_shelf is None:
            self.logger.error("No shelf found for initial angle!")
            return False
        
        self.current_shelf_index = best_index
        self.current_shelf = best_shelf
        self.shelf_navigation_active = True
        
        self.logger.info(f"Selected FIRST shelf {best_index + 1} (angle error: {min_angle_error:.1f}°)")
        
        # Navigate to major_axis for object detection
        major_x, major_y, major_angle = best_shelf['major_axis']
        goal = self.create_goal_from_world_coord(major_x, major_y, major_angle)
        self.at_major_axis = True
        
        return self.send_goal_from_world_pose(goal)
    
    def get_frontiers_for_space_exploration(self, map_array):#✅
        """Identifies frontiers for space exploration.

        Args:
            map_array: 2D numpy array representing the map.

        Returns:
            frontiers: List of tuples representing frontier coordinates.
        """
        frontiers = []
        for y in range(1, map_array.shape[0] - 1):
            for x in range(1, map_array.shape[1] - 1):
                if map_array[y, x] == -1:  # Unknown space and not visited.
                    neighbors_complete = [
                        (y, x - 1),
                        (y, x + 1),
                        (y - 1, x),
                        (y + 1, x),
                        (y - 1, x - 1),
                        (y + 1, x - 1),
                        (y - 1, x + 1),
                        (y + 1, x + 1)
                    ]

                    near_obstacle = False
                    for ny, nx in neighbors_complete:
                        if map_array[ny, nx] > 0:  # Obstacles.
                            near_obstacle = True
                            break
                    if near_obstacle:
                        continue

                    neighbors_cardinal = [
                        (y, x - 1),
                        (y, x + 1),
                        (y - 1, x),
                        (y + 1, x),
                    ]

                    for ny, nx in neighbors_cardinal:
                        if map_array[ny, nx] == 0:  # Free space.
                            frontiers.append((ny, nx))
                            break

        return frontiers
    


    def publish_debug_image(self, publisher, image):#✅
        """Publishes images for debugging purposes.

        Args:
            publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
            image: Image given by an n-dimensional numpy array.

        Returns:
            None
        """
        if image.size:
            message = CompressedImage()
            _, encoded_data = cv2.imencode('.jpg', image)
            message.format = "jpeg"
            message.data = encoded_data.tobytes()
            publisher.publish(message)

    def camera_image_callback(self, message):
        """Callback function to handle incoming camera images.

        Args:
            message: ROS2 message of the type sensor_msgs.msg.CompressedImage.

        Returns:
            None
        """
        """Handle camera images for QR scanning during shelf navigation"""
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Only scan QR codes when at minor axis (short viewpoint) during shelf navigation
        if self.shelf_navigation_active and self.qr_scanning:
            try:
                decoded_objs = decode(image)
                
                if decoded_objs:
                    qr_data = decoded_objs[0].data.decode("utf-8")
                    self.qr_data= qr_data  
                    self.qr_code_str = qr_data
                    self.logger.info(f"QR Code detected: {qr_data}")
                    
                    # Parse QR code for next shelf angle
                    try:
                        parts = qr_data.split('_')
                        if len(parts) >= 2:
                            next_shelf_angle = float(parts[1])
                            self.initial_angle = next_shelf_angle
                            self.logger.info(f" Next shelf angle: {next_shelf_angle}°")
                            
                            # Stop QR scanning once we get the data
                            self.qr_scanning = False
                            
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"QR parsing failed: {e}")
                        
            except Exception as e:
                self.logger.error(f"QR detection failed: {e}")


        # Optional line for visualizing image on foxglove.
        # self.publish_debug_image(self.publisher_qr_decode, image)

    def complete_current_shelf_callback(self):
        """Timer callback to complete current shelf"""
        if self.qr_timer is not None:
            self.qr_timer.destroy()
            self.qr_timer = None
        self.complete_current_shelf()


    def complete_current_shelf(self):
        """Complete current shelf processing and move to next shelf"""
        if not self.current_shelf:
            self.logger.error("No current shelf to complete")
            return
            
        # Mark current shelf as visited
        self.current_shelf['visited'] = True
        
        # Publish shelf data
        self.publish_shelf_data()
        
        # Log completion
        self.logger.info(f"Completed shelf {self.current_shelf_index + 1}")
        self.logger.info(f"Final objects: {self.object_names}")
        self.logger.info(f"Final counts: {self.object_count}")
        self.logger.info(f"QR code: {self.qr_data}")
        
        # Reset state flags
        self.at_major_axis = False
        self.qr_scanning = False
        
        # Move to next shelf
        self.navigate_to_next_shelf()

    def navigate_to_next_shelf(self):
        """Navigate to next shelf using QR-decoded angle from origin (0,0)"""
        if self.qr_data == "Empty":
            self.logger.info("No QR data available - workflow complete!")
            self.shelf_navigation_active = False
            return
        qr_angle = self.initial_angle
        # Find next shelf using QR-decoded angle
        best_shelf = None
        best_index = -1
        min_angle_error = float('inf')
        
        self.logger.info(f"Looking for next shelf using QR angle: {qr_angle}°")
        
        for i, shelf in enumerate(self.shelves_final):
            if shelf.get('visited', False):
                continue
                
            cx, cy = shelf['center_world']
            
            # Calculate angle from world origin (0,0) to shelf center
            angle_to_shelf = math.atan2(cy - 0.0, cx - 0.0)
            angle_deg = (math.degrees(angle_to_shelf) + 360) % 360
            
            # Match against QR-decoded angle
            error = abs((angle_deg - self.initial_angle + 180) % 360 - 180)
            
            self.logger.info(f"  Shelf {i+1}: angle={angle_deg:.1f}°, error={error:.1f}°")
            
            if error < min_angle_error:
                min_angle_error = error
                best_shelf = shelf
                best_index = i
        
        if best_shelf is None:
            self.logger.info("All shelves visited - workflow complete!")
            self.shelf_navigation_active = False
            return
        
        self.current_shelf_index = best_index
        self.current_shelf = best_shelf
        
        

        self.logger.info(f"Selected NEXT shelf {best_index + 1} (angle error: {min_angle_error:.1f}°)")
        
        # Navigate to long viewpoint for object detection
        major_x, major_y, major_angle = best_shelf['major_axis']
        goal = self.create_goal_from_world_coord(major_x, major_y, major_angle)
        self.at_major_axis = True
        
        self.send_goal_from_world_pose(goal)



    def merge_object_detections(self, prev_names, prev_counts, curr_names, curr_counts):
        merged_dict = {}

        # Normalize previous and current names (lowercase and strip)
        prev_dict = {name.strip().lower(): count for name, count in zip(prev_names, prev_counts)}
        curr_dict = {name.strip().lower(): count for name, count in zip(curr_names, curr_counts)}

        # Union all object names
        all_object_names = set(prev_dict.keys()).union(set(curr_dict.keys()))

        # Allowed object list
        allowed_objects = {
            "banana", "zebra", "teddy bear", "car",
            "potted plant", "cup", "clock", "horse"
        }

        for name in all_object_names:
            if name not in allowed_objects:
                continue  # Skip objects not in allowed list

            prev_count = prev_dict.get(name, 0)
            curr_count = curr_dict.get(name, 0)

            # Merge rule
            merged_dict[name] = max(prev_count, curr_count)

        # Convert to lists and preserve original casing from current/previous names
        merged_names = []
        merged_counts = []

        for name, count in merged_dict.items():
            original_name = None
            for n in curr_names + prev_names:
                if n.strip().lower() == name:
                    original_name = n.strip()
                    break
            merged_names.append(original_name or name)
            merged_counts.append(count)

        return merged_names, merged_counts
    
    def publish_shelf_data(self):
        """Publish complete shelf data with objects and QR code"""
        shelf_message = WarehouseShelf()
        shelf_message.object_name = self.object_names
        shelf_message.object_count = self.object_count
        shelf_message.qr_decoded = self.qr_data
        
        self.publisher_shelf_data.publish(shelf_message)
        
        self.logger.info(f"   Published shelf data:")
        self.logger.info(f"   Objects: {self.object_names}")
        self.logger.info(f"   Counts: {self.object_count}")
        self.logger.info(f"   QR Code: {self.qr_data}")
        
        # Reset for next shelf
        self.object_names = []
        self.object_count = []

    def move_to_minor_axis_callback(self):
        """Timer callback to move to minor axis"""
        if self.object_timer is not None:
            self.object_timer.destroy()
            self.object_timer = None
        self.move_to_minor_axis()

    def move_to_minor_axis(self):
        """move from major axis to minor axis for QR scanning"""
        if not self.current_shelf:
            self.logger.info("Not currently at any shelf...")
            return
        minor_x,minor_y,minor_angle = self.current_shelf['minor_axis']
        self.logger.info(f"Moving to the point ({minor_x:.2f},{minor_y:.2f})")
        goal = self.create_goal_from_world_coord(minor_x,minor_y,minor_angle)
        self.at_major_axis = False
        self.qr_scanning = True

        status = self.send_goal_from_world_pose(goal)

        if status:
            self.logger.info("Successfully send goal to the minor axis for QR scanning...")
        else :
            self.logger.info("Failed to send to goal!!!")

    def cerebri_status_callback(self, message):#✅
        """Callback function to handle cerebri status updates.

        Args:
            message: ROS2 message containing cerebri status.

        Returns:
            None
        """
        if message.mode == 3 and message.arming == 2:
            self.armed = True
        else:
            # Initialize and arm the CMD_VEL mode.
            msg = Joy()
            msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
            msg.axes = [0.0, 0.0, 0.0, 0.0]
            self.publisher_joy.publish(msg)

    def behavior_tree_log_callback(self, message):#✅ 
        """Alternative method for checking goal status.

        Args:
            message: ROS2 message containing behavior tree log.

        Returns:
            None
        """
        for event in message.event_log:
            if (event.node_name == "FollowPath" and
                event.previous_status == "SUCCESS" and
                event.current_status == "IDLE"):
                # self.goal_completed = True
                # self.goal_handle_curr = None
                pass

    def shelf_objects_callback(self, message):
        """Callback function to handle shelf objects updates.

        Args:
            message: ROS2 message containing shelf objects data.

        Returns:
            None
        """
        self.shelf_objects_curr = message
        if self.shelf_navigation_active and self.at_major_axis:
            self.object_names, self.object_count = self.merge_object_detections(self.object_names, self.object_count,
            message.object_name, message.object_count)
            # self.logger.info(f"Updated merged objects: {self.object_names}")
            # self.logger.info(f"Updated merged counts: {self.object_count}")
        # Process the shelf objects as needed.

        # How to send WarehouseShelf messages for evaluation.

        """
        * Example for sending WarehouseShelf messages for evaluation.
            shelf_data_message = WarehouseShelf()

            shelf_data_message.object_name = ["car", "clock"]
            shelf_data_message.object_count = [1, 2]
            shelf_data_message.qr_decoded = "test qr string"

            self.publisher_shelf_data.publish(shelf_data_message)

        * Alternatively, you may store the QR for current shelf as self.qr_code_str.
            Then, add it as self.shelf_objects_curr.qr_decoded = self.qr_code_str
            Then, publish as self.publisher_shelf_data.publish(self.shelf_objects_curr)
            This, will publish the current detected objects with the last QR decoded.
        """

        # Optional code for populating TABLE GUI with detected objects and QR data.
        """
        if PROGRESS_TABLE_GUI:
            shelf = self.shelf_objects_curr
            obj_str = ""
            for name, count in zip(shelf.object_name, shelf.object_count):
                obj_str += f"{name}: {count}\n"

            box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
            box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
            self.table_row_count += 1

            box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
            box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
            self.table_row_count = 0
            self.table_col_count += 1
        """

    def rover_move_manual_mode(self, speed, turn):#✅ 
        """Operates the rover in manual mode by publishing on /cerebri/in/joy.

        Args:
            speed: The speed of the car in float. Range = [-1.0, +1.0];
                   Direction: forward for positive, reverse for negative.
            turn: Steer value of the car in float. Range = [-1.0, +1.0];
                  Direction: left turn for positive, right turn for negative.

        Returns:
            None
        """
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, speed, 0.0, turn]
        self.publisher_joy.publish(msg)



    def cancel_goal_callback(self, future):#✅
        """
        Callback function executed after a cancellation request is processed.

        Args:
            future (rclpy.Future): The future is the result of the cancellation request.
        """
        cancel_result = future.result()
        if cancel_result:
            self.logger.info("Goal cancellation successful.")
            self.cancelling_goal = False  # Mark cancellation as completed (success).
            return True
        else:
            self.logger.error("Goal cancellation failed.")
            self.cancelling_goal = False  # Mark cancellation as completed (failed).
            return False

    def cancel_current_goal(self):#✅
        """Requests cancellation of the currently active navigation goal."""
        if self.goal_handle_curr is not None and not self.cancelling_goal:
            self.cancelling_goal = True  # Mark cancellation in-progress.
            self.logger.info("Requesting cancellation of current goal...")
            cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
            cancel_future.add_done_callback(self.cancel_goal_callback)

    def goal_result_callback(self, future):#✅
        """
        Callback function executed when the navigation goal reaches a final result.

        Args:
            future (rclpy.Future): The future that is result of the navigation action.
        """
        status = future.result().status
        # NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

        if status == GoalStatus.STATUS_SUCCEEDED:
            self.logger.info("Goal completed successfully!")
        else:
            self.logger.warn(f"Goal failed with status: {status}")

        if self.shelf_navigation_active:
            if self.at_major_axis:
                if self.object_timer is not None:
                    self.object_timer.destroy()
                self.logger.info("Identifying Shelf objects for 5s...")
                self.object_timer = self.create_timer(5.0,self.move_to_minor_axis_callback)
            else:
                if self.qr_timer is not None:
                    self.qr_timer.destroy()
                self.logger.info("Scanning the QR for next 5s interval...")
                self.qr_timer = self.create_timer(5.0,self.complete_current_shelf_callback)
        else :
            self.logger.warn("Goal failed !!!")

        self.goal_completed = True  # Mark goal as completed.
        self.goal_handle_curr = None  # Clear goal handle.

    def goal_response_callback(self, future):#✅
        """
        Callback function executed after the goal is sent to the action server.

        Args:
            future (rclpy.Future): The future that is server's response to goal request.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.logger.warn('Goal rejected :(')
            self.goal_completed = True  # Mark goal as completed (rejected).
            self.goal_handle_curr = None  # Clear goal handle.
        else:
            self.logger.info('Goal accepted :)')
            self.goal_completed = False  # Mark goal as in progress.
            self.goal_handle_curr = goal_handle  # Store goal handle.

            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(self.goal_result_callback)

    def goal_feedback_callback(self, msg):#✅
        """
        Callback function to receive feedback from the navigation action.

        Args:
            msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
        """
        distance_remaining = msg.feedback.distance_remaining
        number_of_recoveries = msg.feedback.number_of_recoveries
        navigation_time = msg.feedback.navigation_time.sec
        estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

        self.logger.debug(f"Recoveries: {number_of_recoveries}, "
                  f"Navigation time: {navigation_time}s, "
                  f"Distance remaining: {distance_remaining:.2f}, "
                  f"Estimated time remaining: {estimated_time_remaining}s")

        if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
            self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
            self.cancel_current_goal()  # Unblock by discarding the current goal.

    def send_goal_from_world_pose(self, goal_pose):#✅
        """
        Sends a navigation goal to the Nav2 action server.

        Args:
            goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

        Returns:
            bool: True if the goal was successfully sent, False otherwise.
        """
        if not self.goal_completed or self.goal_handle_curr is not None:
            return False

        self.goal_completed = False  # Starting a new goal.

        goal = NavigateToPose.Goal()
        goal.pose = goal_pose

        if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
            self.logger.error('NavigateToPose action server not available!')
            return False

        # Send goal asynchronously (non-blocking).
        goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
        goal_future.add_done_callback(self.goal_response_callback)

        return True



    def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:#✅
        """Helper function to get map origin and resolution."""
        if map_info:
            origin = map_info.origin
            resolution = map_info.resolution
            return resolution, origin.position.x, origin.position.y
        else:
            return None

    def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
                       -> Tuple[float, float]:#✅
        """Converts map coordinates to world coordinates."""
        if map_info:
            resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
            world_x = (map_x + 0.5) * resolution + origin_x
            world_y = (map_y + 0.5) * resolution + origin_y
            return (world_x, world_y)
        else:
            return (0.0, 0.0)

    def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
                       -> Tuple[int, int]:#✅
        """Converts world coordinates to map coordinates."""
        if map_info:
            resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
            map_x = int((world_x - origin_x) / resolution)
            map_y = int((world_y - origin_y) / resolution)
            return (map_x, map_y)
        else:
            return (0, 0)

    def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:#✅
        """Helper function to create a Quaternion from a yaw angle."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = sy
        q.w = cy
        return q

    def create_yaw_from_vector(self, dest_x: float, dest_y: float,
                   source_x: float, source_y: float) -> float:#✅
        """Calculates the yaw angle from a source to a destination point.
            NOTE: This function is independent of the type of map used.

            Input: World coordinates for destination and source.
            Output: Angle (in radians) with respect to x-axis.
        """
        delta_x = dest_x - source_x
        delta_y = dest_y - source_y
        yaw = math.atan2(delta_y, delta_x)

        return yaw

    def create_goal_from_world_coord(self, world_x: float, world_y: float,
                     yaw: Optional[float] = None) -> PoseStamped:#✅
        """Creates a goal PoseStamped from world coordinates.
            NOTE: This function is independent of the type of map used.
        """
        goal_pose = PoseStamped()
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_pose.header.frame_id = self._frame_id

        goal_pose.pose.position.x = world_x
        goal_pose.pose.position.y = world_y

        if yaw is None and self.pose_curr is not None:
            # Calculate yaw from current position to goal position.
            source_x = self.pose_curr.pose.pose.position.x
            source_y = self.pose_curr.pose.pose.position.y
            yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
        elif yaw is None:
            yaw = 0.0
        else:  # No processing needed; yaw is supplied by the user.
            pass

        goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

        pose = goal_pose.pose.position
        print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
        return goal_pose

    def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
                       yaw: Optional[float] = None) -> PoseStamped:#✅
        """Creates a goal PoseStamped from map coordinates."""
        world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

        return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
    rclpy.init(args=args)

    warehouse_explore = WarehouseExplore()

    if PROGRESS_TABLE_GUI:
        gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
        gui_thread.start()

    rclpy.spin(warehouse_explore)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    warehouse_explore.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()