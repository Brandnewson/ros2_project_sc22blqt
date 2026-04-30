import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
from math import sin, cos
import signal


BLUE_CLOSE_AREA = 70000

WAYPOINTS = [
    (-1.0, -3.7,  0.0),     # centre - establish view of red
    (-5.0, -1.0,  1.57),    # near red
    (-3.0, -5.0, -1.57),    # transition - should expose green and start showing blue
    (-3.0, -7.0,  0.0),     # near blue (loop will break before reaching this)
]

class ProjectNode(Node):

    def __init__(self):
        super().__init__('project_node')

        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.red_detected   = False
        self.green_detected = False
        self.blue_detected  = False
        self.blue_close     = False
        self.blue_area      = 0

        self.sensitivity = 10
        self.stop_msg = Twist()
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.callback, 10)

        self._goal_handle = None
        self._goal_done = False

        self.get_logger().info('ProjectNode initialised.')

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
            return

        hsv_red_lower1  = np.array([0,                       100, 100])
        hsv_red_upper1  = np.array([self.sensitivity,        255, 255])
        hsv_red_lower2  = np.array([180 - self.sensitivity,  100, 100])
        hsv_red_upper2  = np.array([180,                     255, 255])
        hsv_green_lower = np.array([60 - self.sensitivity,   100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity,   255, 255])
        hsv_blue_lower  = np.array([120 - self.sensitivity,  100, 100])
        hsv_blue_upper  = np.array([120 + self.sensitivity,  255, 255])

        Hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        red_mask1  = cv2.inRange(Hsv_image, hsv_red_lower1, hsv_red_upper1)
        red_mask2  = cv2.inRange(Hsv_image, hsv_red_lower2, hsv_red_upper2)
        red_mask   = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(Hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask  = cv2.inRange(Hsv_image, hsv_blue_lower, hsv_blue_upper)

        red_only   = cv2.bitwise_and(image, image, mask=red_mask)
        green_only = cv2.bitwise_and(image, image, mask=green_mask)
        blue_only  = cv2.bitwise_and(image, image, mask=blue_mask)

        self.red_detected   = False
        self.green_detected = False
        self.blue_detected  = False
        self.blue_close     = False
        self.blue_area      = 0

        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(red_contours) > 0:
            c = max(red_contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 300:
                self.red_detected = True
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 2)

        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(green_contours) > 0:
            c = max(green_contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 300:
                self.green_detected = True
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)

        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(blue_contours) > 0:
            c = max(blue_contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 300:
                self.blue_detected = True
                self.blue_area = area
                (x, y), radius = cv2.minEnclosingCircle(c)
                cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 2)
                if area > BLUE_CLOSE_AREA:
                    self.blue_close = True

        cv2.namedWindow('camera_Feed',     cv2.WINDOW_NORMAL)
        cv2.namedWindow('Red Detection',   cv2.WINDOW_NORMAL)
        cv2.namedWindow('Green Detection', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Blue Detection',  cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed',     image)
        cv2.imshow('Red Detection',   red_only)
        cv2.imshow('Green Detection', green_only)
        cv2.imshow('Blue Detection',  blue_only)
        cv2.resizeWindow('camera_Feed',     320, 240)
        cv2.resizeWindow('Red Detection',   320, 240)
        cv2.resizeWindow('Green Detection', 320, 240)
        cv2.resizeWindow('Blue Detection',  320, 240)
        cv2.waitKey(3)

    def send_goal(self, x, y, yaw):
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.orientation.z = sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = cos(yaw / 2.0)

        self.get_logger().info(f'Sending goal: ({x:.2f}, {y:.2f})')
        self.action_client.wait_for_server()

        self._goal_done = False
        self._goal_handle = None
        self._send_future = self.action_client.send_goal_async(goal_msg)
        self._send_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.get_logger().warn('Goal rejected!')
            self._goal_done = True
            return
        result_future = self._goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        self._goal_done = True
        status = future.result().status
        if status == 4:
            self.get_logger().info('Goal reached!')
        else:
            self.get_logger().warn(f'Goal ended with status {status}.')

    def cancel_goal(self):
        if self._goal_handle is not None:
            self._goal_handle.cancel_goal_async()

    def stop(self):
        self.publisher.publish(self.stop_msg)


def wait_for_goal_or_blue(node, timeout=90.0):
    t0 = time.time()
    while node._goal_handle is None and rclpy.ok():
        if time.time() - t0 > 5.0:
            return 'no_handle'
        time.sleep(0.1)
    while not node._goal_done and rclpy.ok():
        if node.blue_close:
            node.cancel_goal()
            return 'blue_close'
        if time.time() - t0 > timeout:
            node.cancel_goal()
            return 'timeout'
        time.sleep(0.1)
    return 'done'


def main():
    global node

    def signal_handler(sig, frame):
        node.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    node = ProjectNode()
    signal.signal(signal.SIGINT, signal_handler)

    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()

    time.sleep(3)

    ever_red = ever_green = ever_blue = False

    try:
        for (x, y, theta) in WAYPOINTS:
            if node.blue_close:
                break

            node.send_goal(x, y, theta)
            result = wait_for_goal_or_blue(node, timeout=90.0)

            ever_red   = ever_red   or node.red_detected
            ever_green = ever_green or node.green_detected
            ever_blue  = ever_blue  or node.blue_detected

            if result == 'blue_close':
                break

        for _ in range(20):
            node.publisher.publish(node.stop_msg)
            time.sleep(0.05)

        ever_red   = ever_red   or node.red_detected
        ever_green = ever_green or node.green_detected
        ever_blue  = ever_blue  or node.blue_detected

        node.get_logger().info(
            f'Task complete. Red: {ever_red}, Green: {ever_green}, Blue: {ever_blue}, blue_close: {node.blue_close}'
        )

        while rclpy.ok():
            time.sleep(1)

    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()