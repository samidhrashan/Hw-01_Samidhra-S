"""
02_verify_skew_properties.py -- HW1 Part 2, Task 2

Verify the rotation identities during a MuJoCo simulation
with time-varying angular velocity.

Identity 1:
    R (v x w) = (R v) x (R w)

Identity 2:
    R hat(omega) R^T = hat(R omega)
"""

import numpy as np
import mujoco

from utils import hat, get_body_orientation, is_close_to_identity


MODEL_PATH = "../model/asymmetric_body.xml"

# Number of random tests at each logged time step
N_CHECKS_PER_STEP = 5

# Number of time points at which we check the identities
N_LOGGED_STEPS = 5

# Number of MuJoCo simulation steps between checks
STEPS_BETWEEN_LOGS = 200


# ============================================================
# TIME-VARYING ANGULAR VELOCITY
# ============================================================

def time_varying_angular_velocity(t):
    """
    Return angular velocity omega(t) in rad/s.

    The angular velocity changes continuously with simulation time.
    """

    wx = 2.0 * np.sin(t)

    wy = 1.5 * np.cos(0.7 * t)

    wz = 1.0 + 0.5 * np.sin(0.5 * t)

    return np.array([wx, wy, wz])


# ============================================================
# CHECK THE TWO ROTATION IDENTITIES
# ============================================================

def check_identities(R, rng):
    """
    Check:

        R @ (v x w) = (R @ v) x (R @ w)

        R @ hat(omega) @ R.T = hat(R @ omega)

    using random vectors.

    Returns:
        maximum residual for identity 1,
        maximum residual for identity 2.
    """

    max_residual_cross = 0.0
    max_residual_skew = 0.0

    for _ in range(N_CHECKS_PER_STEP):

        # ----------------------------------------------------
        # Generate random vectors
        # ----------------------------------------------------

        v = rng.normal(size=3)
        w = rng.normal(size=3)
        omega = rng.normal(size=3)

        # ====================================================
        # IDENTITY 1
        #
        # R (v x w) = (Rv) x (Rw)
        # ====================================================

        left_cross = R @ np.cross(v, w)

        right_cross = np.cross(
            R @ v,
            R @ w
        )

        residual_cross = np.linalg.norm(
            left_cross - right_cross
        )

        max_residual_cross = max(
            max_residual_cross,
            residual_cross
        )

        # ====================================================
        # IDENTITY 2
        #
        # R hat(omega) R^T = hat(R omega)
        # ====================================================

        left_skew = R @ hat(omega) @ R.T

        right_skew = hat(R @ omega)

        residual_skew = np.linalg.norm(
            left_skew - right_skew
        )

        max_residual_skew = max(
            max_residual_skew,
            residual_skew
        )

    return max_residual_cross, max_residual_skew


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load MuJoCo model
    # --------------------------------------------------------

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    data = mujoco.MjData(model)

    # Random number generator
    rng = np.random.default_rng(seed=0)

    # --------------------------------------------------------
    # Print header
    # --------------------------------------------------------

    print()
    print("==========================================================")
    print("        HW1 TASK 2 - SKEW-SYMMETRIC IDENTITIES")
    print("==========================================================")
    print()

    print("Using TIME-VARYING angular velocity:")
    print("omega(t) = [2 sin(t), 1.5 cos(0.7t), 1 + 0.5 sin(0.5t)]")
    print()

    print(
        f"{'step':>5} "
        f"{'t (s)':>8} "
        f"{'omega_x':>10} "
        f"{'omega_y':>10} "
        f"{'omega_z':>10} "
        f"{'cross residual':>18} "
        f"{'skew residual':>18}"
    )

    print("-" * 95)

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    all_results = []

    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    for log_i in range(N_LOGGED_STEPS):

        # ----------------------------------------------------
        # Advance simulation using time-varying angular velocity
        # ----------------------------------------------------

        for _ in range(STEPS_BETWEEN_LOGS):

            t = data.time

            # Calculate omega(t)
            angular_velocity = time_varying_angular_velocity(t)

            # Put angular velocity into free-joint qvel
            data.qvel[3:6] = angular_velocity

            # Advance MuJoCo simulation
            mujoco.mj_step(model, data)

        # ----------------------------------------------------
        # Get current rotation matrix R(t)
        # ----------------------------------------------------

        R = get_body_orientation(data)

        # ----------------------------------------------------
        # Check that R is a valid rotation matrix
        # ----------------------------------------------------

        assert is_close_to_identity(
            R @ R.T,
            tol=1e-6
        ), "R is not orthonormal!"

        # ----------------------------------------------------
        # Angular velocity at current time
        # ----------------------------------------------------

        omega_current = time_varying_angular_velocity(data.time)

        # ----------------------------------------------------
        # Check both identities
        # ----------------------------------------------------

        resid_cross, resid_skew = check_identities(
            R,
            rng
        )

        # ----------------------------------------------------
        # Save results
        # ----------------------------------------------------

        all_results.append([
            log_i,
            data.time,
            omega_current[0],
            omega_current[1],
            omega_current[2],
            resid_cross,
            resid_skew
        ])

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"{log_i:5d} "
            f"{data.time:8.3f} "
            f"{omega_current[0]:10.4f} "
            f"{omega_current[1]:10.4f} "
            f"{omega_current[2]:10.4f} "
            f"{resid_cross:18.3e} "
            f"{resid_skew:18.3e}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    all_results = np.array(all_results)

    np.savetxt(
        "task2_residuals.csv",
        all_results,
        delimiter=",",
        header=(
            "step,time,"
            "omega_x,omega_y,omega_z,"
            "cross_residual,skew_residual"
        ),
        comments=""
    )

    # ========================================================
    # FIND WORST RESIDUAL
    # ========================================================

    worst_cross = np.max(all_results[:, 5])

    worst_skew = np.max(all_results[:, 6])

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("==========================================================")
    print("                       SUMMARY")
    print("==========================================================")

    print(
        f"Worst cross-product residual : "
        f"{worst_cross:.3e}"
    )

    print(
        f"Worst skew-matrix residual   : "
        f"{worst_skew:.3e}"
    )

    print()
    print("Results saved to:")
    print("task2_residuals.csv")

    print()
    print("The residuals should be very small, near")
    print("floating-point machine precision.")

    print()
    print("A small non-zero residual does not mathematically")
    print("prove the identities. It only provides numerical")
    print("evidence that the identities hold.")

    print()
    print("The small residual comes from finite-precision")
    print("floating-point arithmetic and numerical computation.")

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
