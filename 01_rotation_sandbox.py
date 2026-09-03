import time
import numpy as np
import mujoco
import mujoco.viewer

from utils import Rx, Ry, Rz, ELEMENTARY_ROTATIONS, set_body_orientation

MODEL_PATH = "../model/asymmetric_body.xml"

# ============================================================
# ROTATION SEQUENCE
# ============================================================
# Change "current" to "fixed" to record the other case.

rotation_sequence = [
    ("z", np.deg2rad(90), "fixed"),
    ("x", np.deg2rad(90), "fixed"),
]


# ============================================================
# ROTATION COMPOSITION
# ============================================================

def compose_sequence(sequence):
    """
    Calculate the final rotation matrix.

    current -> rotation about the current/body frame
    fixed   -> rotation about the fixed/space frame
    """

    R = np.eye(3)

    for axis, angle, frame in sequence:

        R_step = ELEMENTARY_ROTATIONS[axis](angle)

        if frame == "current":
            # Body/current frame rotation
            R = R @ R_step

        elif frame == "fixed":
            # Space/fixed frame rotation
            R = R_step @ R

        else:
            raise ValueError(
                f"Unknown frame specification: '{frame}'"
            )

    return R


# ============================================================
# ANIMATION SETTINGS
# ============================================================

FPS = 60

# Time before rotation starts
INITIAL_PAUSE = 5.0

# Time taken for EACH rotation
ROTATION_TIME = 10.0

# Pause between rotations
BETWEEN_ROTATIONS =  1.0
# Time to hold final orientation
FINAL_PAUSE = 3.0


# ============================================================
# ANIMATE ONE ROTATION SEQUENCE
# ============================================================

def animate_sequence(data, model, viewer):

    # Start from identity orientation
    R_curr = np.eye(3)

    set_body_orientation(data, R_curr)
    mujoco.mj_forward(model, data)
    viewer.sync()

    print()
    print("==========================================")
    print("Starting animation in 3 seconds...")
    print("==========================================")
    print()

    time.sleep(INITIAL_PAUSE)

    # --------------------------------------------------------
    # Apply each rotation
    # --------------------------------------------------------

    for axis, total_angle, frame in rotation_sequence:

        print(
            f"Applying {axis.upper()} rotation "
            f"{np.rad2deg(total_angle):.1f} degrees "
            f"about {frame} frame"
        )

        steps = int(ROTATION_TIME * FPS)

        for i in range(1, steps + 1):

            if not viewer.is_running():
                return

            # Smoothly increase angle from 0 to total_angle
            angle_step = total_angle * (i / steps)

            R_step = ELEMENTARY_ROTATIONS[axis](angle_step)

            # ------------------------------------------------
            # Current/body frame
            # ------------------------------------------------
            if frame == "current":
                R_anim = R_curr @ R_step

            # ------------------------------------------------
            # Fixed/space frame
            # ------------------------------------------------
            elif frame == "fixed":
                R_anim = R_step @ R_curr

            else:
                raise ValueError(
                    f"Unknown frame specification: '{frame}'"
                )

            # Update MuJoCo body orientation
            set_body_orientation(data, R_anim)

            mujoco.mj_forward(model, data)

            viewer.sync()

            time.sleep(1.0 / FPS)

        # Save the final orientation of this rotation
        R_curr = R_anim

        # Small pause before next rotation
        time.sleep(BETWEEN_ROTATIONS)

    # --------------------------------------------------------
    # Hold final orientation
    # --------------------------------------------------------

    print()
    print("==========================================")
    print("Final orientation reached.")
    print("Holding final pose...")
    print("==========================================")
    print()

    time.sleep(FINAL_PAUSE)


# ============================================================
# MAIN
# ============================================================

def main():

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    # Create simulation data
    data = mujoco.MjData(model)

    # Calculate final rotation matrix
    R_final = compose_sequence(rotation_sequence)

    print()
    print("==========================================")
    print("        HW1 TASK 1 - ROTATION")
    print("==========================================")

    print()
    print("Rotation sequence:")
    for axis, angle, frame in rotation_sequence:
        print(
            f"  {axis.upper()} : "
            f"{np.rad2deg(angle):.1f} degrees : "
            f"{frame} frame"
        )

    print()
    print("Final rotation matrix:")
    print(R_final)

    print()
    print("Opening MuJoCo viewer...")
    print()

    # Open MuJoCo viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Run animation ONCE
        animate_sequence(data, model, viewer)

        # Keep viewer open after animation
        print("Animation finished.")
        print("Close the MuJoCo window when you are done.")

        while viewer.is_running():

            viewer.sync()

            time.sleep(1.0 / FPS)


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
