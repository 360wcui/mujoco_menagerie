import mujoco
import mujoco.viewer
import numpy as np

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
pids = [PID(1, 0.5, 0.2) for _ in range(7)]

# Target joint positions (radians) for the arm only
target_qpos = np.array([1, -0.5, 0, -1.5, 0, 1.0, 0.5])

# Identify the indices of the 7 main joints in qpos
# The first 7 revolute joints of the arm are usually the first 7 in the actuator mapping
arm_joint_ids = [model.actuator_trnid[i,0] for i in range(7)]  # joints 0-6
arm_qpos_indices = [model.jnt_qposadr[j] for j in arm_joint_ids]

dt = model.opt.timestep

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        torques = []
        for i, qpos_idx in enumerate(arm_qpos_indices):
            current = data.qpos[qpos_idx]
            torques.append(pids[i](target_qpos[i], current, dt))

        # Apply torques to first 7 actuators (the arm)
        data.ctrl[:7] = torques

        mujoco.mj_step(model, data)
        viewer.sync()