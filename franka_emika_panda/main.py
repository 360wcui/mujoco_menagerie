import mujoco.viewer
import numpy as np

from pid import *

# Load model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# PID controllers for 7 main arm joints
# arm_pids = [
#     PID(30, 0.1, 0.5),  # joint 1 Base rotation — rotates entire arm left/right
#     PID(1.4, 0.1, 0.3),  # joint 2 Shoulder pitch — lifts arm up/down
#     PID(1.1, 0.1, 0.2),# joint 3 Elbow yaw — rotates forearm left/right
#     PID(1.3, 0.1, 0.3),  # joint 4 Elbow pitch — bends arm forward/backward
#     PID(1.1, 0.1, 0.4),# joint 5 Wrist yaw — rotates wrist left/right
#     PID(1.3, 0.1, 0.3), # joint 6 Wrist pitch — moves wrist up/down
#     PID(1.1, 0.1, 0.3)  # joint 7 Wrist roll — rotates end-effector around its axis
# ]
arm_pids = [
    PID(30, 2, 5),
    PID(25, 1, 5),
    PID(20, 1, 5),
    PID(15, 1, 3),
    PID(10, 1, 3),
    PID(8, 0.6, 2),
    PID(5, 0.2, 2)
]
finger_pid = PID(100, 1, 5)
max_torque = 500

# --- Arm joint indices ---
arm_qpos_indices = [0,1,2,3,4,5,6]

# --- Waypoints in 3D ---
waypoints = [
    (0.6, 0, 0.3),
    (0.6, 0, 0.2),
    (0.6, 0, 0.5),
    (-0.6, 0, 0.5)
]

# Finger open/close
gripper_open = 0.8  # meters
gripper_closed = 0.0

dt = model.opt.timestep

# Arm qpos indices
arm_qpos_indices = [model.jnt_qposadr[i] for i in range(7)]

ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
block_pos = np.array([0.6, 0.0, 0.1])
table_pos = np.array([0, 0.6, 0.1])

hover_above = [block_pos[0], block_pos[1], block_pos[2] + 0.2]
hover_close = [block_pos[0], block_pos[1], block_pos[2] + 0.05]
hover_below = [block_pos[0], block_pos[1], block_pos[2] - 0.05]
lift = [block_pos[0], block_pos[1], block_pos[2] + 0.4]
place_above = [table_pos[0], table_pos[1], table_pos[2] + 0.3]


sequence = [
    (hover_above, gripper_open),
    (hover_below, gripper_open),
    (hover_below, gripper_closed),
    (lift, gripper_closed),
    (place_above, gripper_closed),
    (place_above, gripper_closed),
    (place_above, gripper_open),
]

# ---------------- Simulation ----------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 125
    viewer.cam.elevation = -30
    viewer.cam.distance = 2.2
    viewer.cam.lookat[:] = np.array([0.5, 0, 0.25])

    while viewer.is_running():
        for waypoint, target_finger in sequence:
            # Interpolate to make smooth motion
            q_guess = data.qpos[arm_qpos_indices].copy()
            target_qpos = ik_solve(waypoint, q_guess)
            for _ in range(1500):
                torques = []
                for i, idx in enumerate(arm_qpos_indices):
                    current = data.qpos[idx]
                    torques.append(arm_pids[i](target_qpos[i], current, dt))
                data.ctrl[:7] = torques

                # Finger control (map 0–0.04 m to 0–255)
                data.ctrl[7] = np.clip((target_finger / 0.04) * 500, 0, 500)

                mujoco.mj_step(model, data)
                viewer.sync()

