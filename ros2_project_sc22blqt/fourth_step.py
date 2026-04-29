# Exercise 4 - following a colour (green) and stopping upon sight of another (blue).

#from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class Robot(Node):
    def __init__(self):
        super().__init__('robot')
        
        # Initialise a publisher to publish messages to the robot base
        # We covered which topic receives messages that move the robot in the 3rd Lab Session
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Initialise any flags that signal a colour has been detected (default to false)
        self.green_detected = False
        self.blue_detected = False
        self.move_forwards = False
        self.move_backwards = False

        # Initialise the value you wish to use for sensitivity in the colour detection (10 should be enough)
        self.sensitivity = 10

        # Initialise some standard movement messages such as a simple move forward and a message with all zeroes (stop)
        self.move_forward_msg = Twist()
        self.move_forward_msg.linear.x = 0.2

        self.move_backward_msg = Twist()
        self.move_backward_msg.linear.x = -0.2

        self.stop_msg = Twist()  # all zeros by default

        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning

        # We covered which topic to subscribe to should you wish to receive image data

    def callback(self, data):

        # Convert the received image into a opencv image
        # But remember that you should always wrap a call to this conversion method in an exception handler
        image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL)
        cv2.imshow('camera_Feed', image)
        cv2.resizeWindow('camera_Feed',320,240)
        cv2.waitKey(3)
        

        # Set the upper and lower bounds for the two colours you wish to identify
        # hue value = 0 to 179
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])

        # Convert the rgb image into a hsv image
        Hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Filter out everything but a particular colour using the cv2.inRange() method
        green_mask = cv2.inRange(Hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask  = cv2.inRange(Hsv_image, hsv_blue_lower,  hsv_blue_upper)

        # Apply the mask to the original image using the cv2.bitwise_and() method
        green_only = cv2.bitwise_and(image, image, mask=green_mask)
        blue_only  = cv2.bitwise_and(image, image, mask=blue_mask)

        self.green_detected = False
        self.blue_detected = False
        self.move_forwards = False
        self.move_backwards = False
        
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        blue_contours, _  = cv2.findContours(blue_mask,  cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



        # Filter out everything but a particular colour using the cv2.inRange() method


        # Apply the mask to the original image using the cv2.bitwise_and() method
        # As mentioned on the worksheet the best way to do this is to bitwise and an image with itself and pass the mask to the mask parameter


        # Find the contours that appear within the certain colour mask using the cv2.findContours() method
        # For <mode> use cv2.RETR_LIST for <method> use cv2.CHAIN_APPROX_SIMPLE

        if len(green_contours) > 0:
            # Use the max() method to find the largest contour
            c = max(green_contours, key=cv2.contourArea)

            # Check if the area of the shape you want is big enough to be considered
            # If it is then change the flag for that colour to be True
            if cv2.contourArea(c) > 500:
                self.green_detected = True

                if cv2.contourArea(c) > 50000:
                    # Too close to object, need to move backwards
                    # Set a flag to tell the robot to move backwards when in the main loop
                    self.move_backwards = True
                else:
                    # Too far away from object, need to move forwards
                    # Set a flag to tell the robot to move forwards when in the main loop
                    self.move_forwards = True

        if len(blue_contours) > 0:
            c = max(blue_contours, key=cv2.contourArea)

            # Check if the area is big enough to be considered
            if cv2.contourArea(c) > 5000:
                # Blue detected - set flag to stop
                self.blue_detected = True

        # Show the resultant images you have created. You can show all of them or just the end result if you wish to.
        cv2.imshow('Green Detection', green_only)
        cv2.imshow('Blue Detection', blue_only)
        cv2.imshow('camera_Feed', image)
        cv2.waitKey(3)

    def walk_forward(self):
        for _ in range(30):
            self.publisher.publish(self.move_forward_msg)
            time.sleep(0.01)

    def walk_backward(self):
        for _ in range(30):
            self.publisher.publish(self.move_backward_msg)
            time.sleep(0.01)

    def stop(self):
        self.publisher.publish(self.stop_msg)

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():
    def signal_handler(sig, frame):
        robot.stop()
        rclpy.shutdown()

    # Instantiate your class
    # And rclpy.init the entire node
    rclpy.init(args=None)
    robot = Robot()
    


    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            if robot.blue_detected:
                robot.stop()
            elif robot.green_detected:
                if robot.move_backwards:
                    robot.walk_backward()
                else:
                    robot.walk_forward()
            else:
                robot.stop()
    except ROSInterruptException:
        pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
