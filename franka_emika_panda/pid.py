from spatialmath import SE3
import roboticstoolbox as rtb
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

def ik_solve(target_xyz, q_guess=None):
    panda = rtb.models.Panda()
    print("target xyz", target_xyz)
    T = SE3(target_xyz[0], target_xyz[1], target_xyz[2]) * SE3.Rx(np.pi)  # end-effector at target
    sol = panda.ikine_LM(T, q0=q_guess)  # only position

    if sol.success:
        return sol.q
    else:
        raise RuntimeError("IK failed for target:", target_xyz)

def solve_ik(model, data, site_name, target_pos, position_only=False, target_quat=None,
             max_iters=2000, tol=1e-3, step_size=0.01):
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
    target_quat : np.array shape (4,), optional
        Desired orientation as [w, x, y, z]. Ignored if position_only=True.
    max_iters : int
        Maximum IK iterations
    tol : float
        Convergence tolerance
    step_size : float
        Step size for gradient descent
    position_only : bool
        If True, solve only for position (ignore orientation).

    Returns
    -------
    qpos : np.array
        Joint positions (7 DOF arm) that reach the target
    """
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if site_id == -1:
        raise ValueError(f"Site '{site_name}' not found in model.")

    nv = model.nv

    for i in range(max_iters):
        mj.mj_forward(model, data)  # update FK

        # --- Position error ---
        current_pos = data.site_xpos[site_id].copy()
        pos_err = target_pos - current_pos

        if position_only:
            err = pos_err
        else:
            # --- Orientation error ---
            current_mat = data.site_xmat[site_id].reshape(3, 3)
            target_rot = R.from_quat([target_quat[1], target_quat[2], target_quat[3], target_quat[0]]).as_matrix()
            rot_err = R.from_matrix(target_rot @ current_mat.T).as_rotvec()

            # Full 6D error
            err = np.concatenate([pos_err, rot_err])

        # Check convergence
        if np.linalg.norm(err) < tol:
            return data.qpos[:7].copy()

        # --- Jacobian ---
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mj.mj_jacSite(model, data, jacp, jacr, site_id)

        if position_only:
            J = jacp  # (3, nv)
        else:
            J = np.vstack([jacp, jacr])  # (6, nv)

        # --- Least squares solve ---
        dq, *_ = np.linalg.lstsq(J, err, rcond=None)

        # --- Apply update ---
        data.qpos[:7] = data.qpos[:7] + dq[:7] * step_size
        mj.mj_forward(model, data)  # update FK after step

    raise RuntimeError("IK did not converge")