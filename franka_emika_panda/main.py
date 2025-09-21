import mujoco.viewer
import numpy as np
import time

# Helper function to compute joint positions for a target
def ik_solve(target_pos, target_quat):
    """
    Solve for joint angles to reach a target Cartesian pose
    target_pos: [x, y, z]
    target_quat: [w, x, y, z]
    """
    # ID of the end-effector
    ee_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "panda_hand")

    # Current joint positions as initial guess
    q_init = data.qpos.copy()

    # Use MuJoCo IK solver
    solved_qpos = mj.mj_kinematics(model, data, target_pos, target_quat, ee_id, q_init)

    return solved_qpos

class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0
        self.prev_error = 0

    def __call__(self, target, current, dt):
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Load model
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)

# PID controllers for 7 main arm joints
arm_pids = [PID(1, 0.5, 0.2) for _ in range(7)]

# Home joint configuration
home_qpos = np.array([0, 0, 0, -1.57, 0, 1.57, -0.7853])

# Finger open/close
gripper_open = 0.04  # meters
gripper_closed = 0.0

dt = model.opt.timestep

# Arm qpos indices
arm_qpos_indices = [model.jnt_qposadr[i] for i in range(7)]

block_z = 0.05      # block height
gripper_offset = 0.1 # distance from wrist to gripper tip

# Above block
pick_above = np.array([0, 0, 0, -1.57, 0, 1.57, -0.7853])
# Block top (z=0.05 + gripper)
pick_down = pick_above + np.array([0, 0, 0, 0.1, 0, 0, 0])
# Lift after grasp
lift = pick_above + np.array([0, 0, 0.2, 0, 0, 0, 0])
# Above table
place_above = pick_above + np.array([0, 0.2, 0.2, 0, 0, 0, 0])
# Place down on table (table z=0.2 + block 0.05)
place_down = place_above + np.array([0, 0, -0.08, 0, 0, 0, 0])

sequence = [
    (pick_above, gripper_open),
    (pick_down, gripper_open),
    (pick_down, gripper_closed),
    (lift, gripper_closed),
    (place_above, gripper_closed),
    (place_down, gripper_closed),
    (place_down, gripper_open),
    (home_qpos, gripper_open)
]

# ---------------- Simulation ----------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 205
    viewer.cam.elevation = -30
    viewer.cam.distance = 4.2
    viewer.cam.lookat[:] = np.array([0.5, 0, 0.25])

    while viewer.is_running():
        for target_qpos, target_finger in sequence:
            # Interpolate to make smooth motion
            for _ in range(100):
                torques = []
                for i, idx in enumerate(arm_qpos_indices):
                    current = data.qpos[idx]
                    torques.append(arm_pids[i](target_qpos[i], current, dt))
                data.ctrl[:7] = torques

                # Finger control (map 0–0.04 m to 0–255)
                data.ctrl[7] = np.clip((target_finger / 0.04) * 255, 0, 255)

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)