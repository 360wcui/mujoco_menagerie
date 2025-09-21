# Panda Joint Names & Motion
# Joint	Name	Axis of rotation	Function / Motion Description
# 1	panda_joint1	Z (vertical)	Base rotation — rotates entire arm left/right
# 2	panda_joint2	Y	Shoulder pitch — lifts arm up/down
# 3	panda_joint3	Z	Elbow yaw — rotates forearm left/right
# 4	panda_joint4	Y	Elbow pitch — bends arm forward/backward
# 5	panda_joint5	Z	Wrist yaw — rotates wrist left/right
# 6	panda_joint6	Y	Wrist pitch — moves wrist up/down
# 7	panda_joint7	Z	Wrist roll — rotates end-effector around its axis

import mujoco.viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
import mujoco as mj

# Helper function to compute joint positions for a target
def solve_ik(model, data, site_name, target_pos, target_quat, max_iters=2000, tol=1e-3, step_size=0.01):
    """
    Solve inverse kinematics for a Panda end-effector.

    Parameters
    ----------
    model : MjModel
    data : MjData
    site_name : str
        Name of the end-effector site
    target_pos : np.array shape (3,)
        Desired position
    target_quat : np.array shape (4,)
        Desired orientation as [w, x, y, z]
    max_iters : int
        Maximum IK iterations
    tol : float
        Convergence tolerance
    step_size : float
        Fraction of the computed dq to apply each iteration

    Returns
    -------
    qpos : np.array
        Joint positions that reach the target
    """
    # Find site ID
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    print("site id", site_id)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found in model.")

    nv = model.nv  # number of DOFs
    print("nv shape", nv)
    qpos = data.qpos.copy()
    print("data.qpos shape", np.shape(data.qpos))
    for i in range(max_iters):
        # Step simulation to update site frames
        mj.mj_forward(model, data)

        # Position error
        current_pos = data.site_xpos[site_id].copy()
        print("current_pos shape", np.shape(current_pos))
        pos_err = target_pos - current_pos  # shape (3,)

        # Orientation error using rotation matrix
        current_mat = data.site_xmat[site_id].reshape(3, 3)  # current orientation
        # MuJoCo uses [w, x, y, z].
        # Scipy Rotation.from_quat expects [x, y, z, w]
        target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()
        rot_err_mat = target_rot @ current_mat.T
        r = R.from_matrix(rot_err_mat)
        rot_err = r.as_rotvec()  # 3-vector axis-angle

        # Combine position + orientation error
        err = np.concatenate([pos_err, rot_err])  # shape (6,)
        print("err shape", np.shape(err))
        print("pos_err shape", np.shape(pos_err))
        print("rot_err shape", np.shape(rot_err))
        # Check convergence
        print(f"Iter {i}, err norm: {np.linalg.norm(err):.4f}")
        if np.linalg.norm(err) < tol:
            return data.qpos[:7].copy()

        # Compute Jacobian
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mj.mj_jacSite(model, data, jacp, jacr, site_id)
        J = np.vstack([jacp, jacr])  # shape (6, nv)

        print(np.shape(J))
        print(np.shape(err))
        # Solve least squares
        dq, *_ = np.linalg.lstsq(J, err, rcond=None)  # dq shape (nv,)
        print("dq shape", np.shape(dq))
        if dq.shape[0] != nv:
            raise RuntimeError(f"dq shape mismatch: {dq.shape} vs nv={nv}")

        # Apply update
        data.qpos[:7] = data.qpos[:7] + dq[:7] * step_size
        mj.mj_forward(model, data)  # update FK after applying step

    raise RuntimeError("IK did not converge")

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
home_qpos = np.array([0, 0, 0, -1.57, 0, 1.57, 0])

# Finger open/close
gripper_open = 0.4  # meters
gripper_closed = 0.0

dt = model.opt.timestep

# Arm qpos indices
arm_qpos_indices = [model.jnt_qposadr[i] for i in range(7)]

block_z = 0.05      # block height
gripper_offset = 0.1 # distance from wrist to gripper tip

# Above block

# --- Block parameters ---
block_pos = np.array([0.6, 0.0, 0.2])
block_height = 0.2
block_top = block_pos[2] + block_height / 2.0  # 0.3

# --- End-effector poses (position + orientation) ---
gripper_clearance = 0.02  # small offset so gripper fingers aren't inside block

pick_above = np.array([block_pos[0], block_pos[1], block_top + 0.5])   # hover
pick_down  = np.array([block_pos[0], block_pos[1], block_top - gripper_clearance])  # touch block
lift       = np.array([block_pos[0], block_pos[1], block_top + 0.5])   # after grasp

# Orientation: gripper pointing down along -Z
# 180° rotation about X axis = quaternion (x,y,z,w) = (1, 0, 0, 0)
ee_quat = np.array([1.0, 0.0, 0.0, 0.0])

# --- Solve IK for each waypoint ---
def compute_pick_sequence(model, data):
    q_pick_above = solve_ik(model, data, "panda_hand", pick_above, ee_quat)
    q_pick_down  = solve_ik(model, data, "panda_hand", pick_down, ee_quat)
    q_lift       = solve_ik(model, data, "panda_hand", lift, ee_quat)
    print("q_pick_above", q_pick_above)
    print("q_pick_down", q_pick_down)
    print("q_lift", q_lift)

    # block_z = 0.2
    # gripper_offset = 0.107  # same as site pos
    # pick_above = np.array([0.6, 0, block_z + gripper_offset + 0.05, 0, np.pi/2, 0, -np.pi/2])
    # pick_down = np.array([0.6, 0, block_z + gripper_offset, 0, np.pi/2, 0, -np.pi/2])

    return q_pick_above, q_pick_down, q_lift

q_pick_above, q_pick_down, q_lift = compute_pick_sequence(model, data)

sequence = [
    (q_pick_above, gripper_open),
    (q_pick_down, gripper_open),
    (q_pick_down, gripper_closed),
    (q_lift, gripper_closed),
    # (place_above, gripper_closed),
    # (place_down, gripper_closed),
    # (place_down, gripper_open),
    # (home_qpos, gripper_open)
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
            for _ in range(1000):
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