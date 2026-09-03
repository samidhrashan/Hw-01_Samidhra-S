"""
HW1 Part 2 - Task 3
ROS 2 TF demonstration.

Keyboard controls:

    C = current/body-frame composition
    F = fixed/space-frame composition
    Q = quit

Rotation sequence:

    Z = 90 degrees
    X = 90 degrees
    Y = 60 degrees

Current frame:
    R_body = R_body @ R_step

Fixed frame:
    R_body = R_step @ R_body
"""

import sys
import threading
import termios
import tty

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


# ============================================================
# ELEMENTARY ROTATION MATRICES
# ============================================================

def Rx(t):
    c = np.cos(t)
    s = np.sin(t)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c]
    ])


def Ry(t):
    c = np.cos(t)
    s = np.sin(t)

    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])


def Rz(t):
    c = np.cos(t)
    s = np.sin(t)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0]
    ])


ELEMENTARY_ROTATIONS = {
    "x": Rx,
    "y": Ry,
    "z": Rz
}


# ============================================================
# SAME ROTATION SEQUENCE AS TASK 1
# ============================================================

STEP_SEQUENCE = [
    ("z", np.deg2rad(90)),
    ("x", np.deg2rad(90)),
    ("y", np.deg2rad(60)),
]


# ============================================================
# ROTATION MATRIX TO QUATERNION
# Output order: x, y, z, w
# ============================================================

def R_to_quat_xyzw(R):

    trace = np.trace(R)

    if trace > 0.0:

        s = 2.0 * np.sqrt(trace + 1.0)

        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s

    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:

        s = 2.0 * np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        )

        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s

    elif R[1, 1] > R[2, 2]:

        s = 2.0 * np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        )

        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s

    else:

        s = 2.0 * np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        )

        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw])

    q = q / np.linalg.norm(q)

    return q[0], q[1], q[2], q[3]


# ============================================================
# ROS 2 NODE
# ============================================================

class Hw01TfBroadcaster(Node):

    def __init__(self):

        super().__init__("hw01_tf_broadcaster")

        # Current or fixed composition
        self.compose_frame = "current"

        # Initial orientation
        self.R_body = np.eye(3)

        # Start at first rotation
        self.step_index = 0

        # Time between rotations
        self.declare_parameter("step_period", 1.0)

        step_period = self.get_parameter(
            "step_period"
        ).value

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer
        self.timer = self.create_timer(
            step_period,
            self.on_timer
        )

        # Keyboard thread
        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            daemon=True
        )

        self.keyboard_thread.start()

        # Startup information
        self.get_logger().info(
            "=========================================="
        )

        self.get_logger().info(
            "HW01 TF Broadcaster started"
        )

        self.get_logger().info(
            "Keyboard controls:"
        )

        self.get_logger().info(
            "  C = CURRENT / body-frame rotation"
        )

        self.get_logger().info(
            "  F = FIXED / space-frame rotation"
        )

        self.get_logger().info(
            "  Q = quit"
        )

        self.get_logger().info(
            "=========================================="
        )


    # ========================================================
    # KEYBOARD LOOP
    # ========================================================

    def keyboard_loop(self):

        old_settings = termios.tcgetattr(sys.stdin)

        try:

            tty.setcbreak(sys.stdin.fileno())

            while rclpy.ok():

                key = sys.stdin.read(1).lower()

                # --------------------------------------------
                # CURRENT FRAME
                # --------------------------------------------

                if key == "c":

                    self.compose_frame = "current"

                    # Reset orientation
                    self.R_body = np.eye(3)

                    # Restart sequence
                    self.step_index = 0

                    self.get_logger().info(
                        "C pressed -> CURRENT frame"
                    )

                    self.get_logger().info(
                        "Sequence reset to identity"
                    )

                # --------------------------------------------
                # FIXED FRAME
                # --------------------------------------------

                elif key == "f":

                    self.compose_frame = "fixed"

                    # Reset orientation
                    self.R_body = np.eye(3)

                    # Restart sequence
                    self.step_index = 0

                    self.get_logger().info(
                        "F pressed -> FIXED frame"
                    )

                    self.get_logger().info(
                        "Sequence reset to identity"
                    )

                # --------------------------------------------
                # QUIT
                # --------------------------------------------

                elif key == "q":

                    self.get_logger().info(
                        "Q pressed -> shutting down"
                    )

                    rclpy.shutdown()

                    break

        finally:

            termios.tcsetattr(
                sys.stdin,
                termios.TCSADRAIN,
                old_settings
            )


    # ========================================================
    # TIMER CALLBACK
    # ========================================================

    def on_timer(self):

        # ----------------------------------------------------
        # FIXED SPACE FRAME
        # ----------------------------------------------------

        self.broadcast_frame(
            "world",
            "space_frame",
            np.eye(3)
        )

        # ----------------------------------------------------
        # GET NEXT ROTATION
        # ----------------------------------------------------

        axis, angle = STEP_SEQUENCE[
            self.step_index % len(STEP_SEQUENCE)
        ]

        R_step = ELEMENTARY_ROTATIONS[axis](angle)

        # ----------------------------------------------------
        # COMPOSE ROTATION
        # ----------------------------------------------------

        if self.compose_frame == "current":

            # Current/body frame
            self.R_body = self.R_body @ R_step

        else:

            # Fixed/space frame
            self.R_body = R_step @ self.R_body

        # ----------------------------------------------------
        # ADVANCE SEQUENCE
        # ----------------------------------------------------

        self.step_index += 1

        # ----------------------------------------------------
        # PRINT STATUS
        # ----------------------------------------------------

        self.get_logger().info(
            f"Step {self.step_index}: "
            f"{axis.upper()} rotation "
            f"{np.rad2deg(angle):.1f} deg | "
            f"Mode = {self.compose_frame}"
        )

        # ----------------------------------------------------
        # BROADCAST BODY FRAME
        # ----------------------------------------------------

        self.broadcast_frame(
            "world",
            "body_frame",
            self.R_body
        )


    # ========================================================
    # BROADCAST TF FRAME
    # ========================================================

    def broadcast_frame(
        self,
        parent,
        child,
        R
    ):

        t = TransformStamped()

        # Timestamp
        t.header.stamp = (
            self.get_clock()
            .now()
            .to_msg()
        )

        # Frame names
        t.header.frame_id = parent
        t.child_frame_id = child

        # ----------------------------------------------------
        # POSITION
        # ----------------------------------------------------

        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0

        if child == "body_frame":

            t.transform.translation.z = 1.0

        else:

            t.transform.translation.z = 0.0

        # ----------------------------------------------------
        # ORIENTATION
        # ----------------------------------------------------

        qx, qy, qz, qw = R_to_quat_xyzw(R)

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        # ----------------------------------------------------
        # SEND TF
        # ----------------------------------------------------

        self.tf_broadcaster.sendTransform(t)


# ============================================================
# MAIN
# ============================================================

def main(args=None):

    rclpy.init(args=args)

    node = Hw01TfBroadcaster()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        if rclpy.ok():

            rclpy.shutdown()

        node.destroy_node()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
