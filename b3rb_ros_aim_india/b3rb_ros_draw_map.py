# # Copyright 2025 NXP

# # Copyright 2016 Open Source Robotics Foundation, Inc.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import rclpy
# from rclpy.node import Node
# from rclpy.parameter import Parameter

# import matplotlib.pyplot as plt
# import matplotlib as mpl
# mpl.rcParams['toolbar'] = 'None'
# import numpy as np
# import math

# from nav_msgs.msg import OccupancyGrid

# import time

# QOS_PROFILE_DEFAULT = 10

# plt.ion()
# plt.axis('off')
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


# class MapVisualizer(Node):
# 	def __init__(self):
# 		super().__init__('map_visualizer')

# 		self.subscription_map = self.create_subscription(
# 			OccupancyGrid,
# 			'/map',
# 			self.map_callback,
# 			QOS_PROFILE_DEFAULT)

# 	def map_callback(self, message):
# 		plt.clf()
# 		width = message.info.width
# 		height = message.info.height

# 		# Convert the occupancy data to a NumPy array.
# 		data = np.array(message.data).reshape((height, width))

# 		# Create an RGB image array.
# 		image = np.zeros((height, width, 3), dtype=np.uint8)

# 		# Assign colors based on occupancy values.
# 		for y in range(height):
# 			for x in range(width):
# 				value = data[y, x]
# 				if value == 0:
# 					image[y, x] = [0, 0, 0]  # Black (free).
# 				elif value == 100:
# 					image[y, x] = [0, 255, 0]  # Green (occupied).
# 				else:  # value == -1
# 					image[y, x] = [127, 127, 127]  # Gray (unknown).

# 		plt.imshow(image)
# 		plt.title("Occupancy Grid (Custom Colors)")
# 		plt.gca().invert_yaxis() #invert the y axis
# 		plt.pause(0.01)
# 		plt.show()


# def main(args=None):
# 	rclpy.init(args=args)

# 	map_visualizer = MapVisualizer()

# 	rclpy.spin(map_visualizer)

# 	# Destroy the node explicitly
# 	# (optional - otherwise it will be done automatically
# 	# when the garbage collector destroys the node object)
# 	map_visualizer.destroy_node()
# 	rclpy.shutdown()


# if __name__ == '__main__':
# 	main()


# Copyright 2025 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.

# Licensed under the Apache License, Version 2.0

import rclpy
from rclpy.node import Node

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'
import numpy as np
import os
import time
from nav_msgs.msg import OccupancyGrid

QOS_PROFILE_DEFAULT = 10

SAVE_INTERVAL = 10.0  # seconds
SAVE_DIR = '/home/sanjay/Downloads/nxp_images_by_yash'
COUNTER_FILE = os.path.join(SAVE_DIR, 'counter.txt')

plt.ion()
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)


class MapVisualizer(Node):
    def __init__(self):
        super().__init__('map_visualizer')

        self.subscription_map = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            QOS_PROFILE_DEFAULT)

        self.latest_image = None
        self.last_save_time = time.time()

        os.makedirs(SAVE_DIR, exist_ok=True)
        self.counter = self.load_counter()

        self.timer = self.create_timer(SAVE_INTERVAL, self.save_image_callback)

    def map_callback(self, message):
        width = message.info.width
        height = message.info.height

        data = np.array(message.data).reshape((height, width))
        image = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                value = data[y, x]
                if value == 0:
                    image[y, x] = [0, 0, 0]
                elif value == 100:
                    image[y, x] = [0, 255, 0]
                else:
                    image[y, x] = [127, 127, 127]

        self.latest_image = image

        plt.clf()
        plt.imshow(image)
        plt.title("Occupancy Grid (Custom Colors)")
        plt.gca().invert_yaxis()
        plt.pause(0.01)
        plt.draw()

    def save_image_callback(self):
        if self.latest_image is not None:
            filename = f"map_{self.counter:05d}.png"
            filepath = os.path.join(SAVE_DIR, filename)

            plt.imsave(filepath, self.latest_image)
            self.get_logger().info(f"Saved map image: {filepath}")
            self.counter += 1
            self.save_counter()

    def load_counter(self):
        if os.path.exists(COUNTER_FILE):
            try:
                with open(COUNTER_FILE, 'r') as f:
                    return int(f.read().strip())
            except Exception:
                return 0
        return 0

    def save_counter(self):
        with open(COUNTER_FILE, 'w') as f:
            f.write(str(self.counter))


def main(args=None):
    rclpy.init(args=args)
    map_visualizer = MapVisualizer()
    rclpy.spin(map_visualizer)
    map_visualizer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
