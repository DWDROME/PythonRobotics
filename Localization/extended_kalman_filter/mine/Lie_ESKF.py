r"""
右不变扩展卡尔曼滤波器（IEKF）在 SE₂(3) 上的演示
================================================

    python lie_eskf.py --scenario spiral --duration 120

相较于经典 ESKF 的关键改进
----------------------------
*   **右不变误差定义**（Barrau 与 Bonnabel, 2017 提出）
*   误差注入后使用 **右雅可比（Jr）映射** 协方差 → 保持滤波器一致性
*   当启用 GNSS 杠杆臂时，测量雅可比矩阵得到修正
*   状态误差向量的排列顺序：δθ（姿态） | δv（速度） | δp（位置） | δb_g（陀螺偏置） | δb_a（加计偏置） | (δs)（刻度因子，可选）
*   支持可选的刻度因子估计与 IMU–GNSS 杠杆臂建模
"""


from __future__ import annotations
import argparse
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple
import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ───────────────────── Lie-group utils ───────────────────────────────────

def skew(v: np.ndarray) -> np.ndarray:
    """Return cross-product matrix ⌈v×⌉ (3×3)."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])

def exp_so3(phi: np.ndarray) -> np.ndarray:
    """Exponential map so(3) → SO(3)."""
    a = norm(phi)
    if a < 1e-12:
        return np.eye(3) + skew(phi)
    A, B = np.sin(a) / a, (1 - np.cos(a)) / (a * a)
    K = skew(phi)
    return np.eye(3) + A * K + B * (K @ K)

def right_jacobian_SO3(phi: np.ndarray) -> np.ndarray:
    """Right Jacobian J_r(φ) (3×3)."""
    a = norm(phi)
    if a < 1e-8:
        return np.eye(3) - 0.5 * skew(phi)
    B = (a - np.sin(a)) / (a ** 3)
    K = skew(phi)
    return np.eye(3) - 0.5 * K + B * (K @ K)

# ───────────────────── Scenario definitions ─────────────────────────────

class ScenarioType(Enum):
    CIRCLE = auto()
    FIGURE_EIGHT = auto()
    SPIRAL = auto()

dataclass_opts = dict()

@dataclass
class GlobalConfig:
    # timing
    dt: float = 0.01
    duration: float = 60.0
    # sensor noise (1σ)
    sigma_gyro: float = 0.002  # rad/s
    sigma_acc: float = 0.05    # m/s²
    sigma_bg_rw: float = 1e-5
    sigma_ba_rw: float = 5e-5
    sigma_gps: float = 0.5
    # bias correlation time
    tau_bg: float = 3600.0
    tau_ba: float = 3600.0
    # gravity
    g_n: np.ndarray = field(default_factory=lambda: np.array([0, 0, -9.81]))
    # earth rotation
    use_earth: bool = True
    omega_ie: float = 7.292115e-5
    # lever-arm
    use_lever: bool = True
    lever_b: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.2]))
    # scale factor
    use_scale: bool = True
    sigma_scale_rw: float = 2e-4
    # gps rate
    gps_period: float = 0.2

CFG = GlobalConfig()
STATE_SIZE = 16 if CFG.use_scale else 15

class Scenario:
    def __init__(self, scenario: ScenarioType):
        self.scenario = scenario

    def true_motion(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        if self.scenario is ScenarioType.CIRCLE:
            omega_b = np.array([0.0, 0.0, 0.02])
            a_n = np.array([0.1*np.cos(0.05*t), 0.1*np.sin(0.05*t), 0.])
            return omega_b, a_n
        if self.scenario is ScenarioType.FIGURE_EIGHT:
            omega_b = np.array([0., 0., 0.04*np.sin(0.05*t)])
            a_n = np.array([0.2*np.sin(0.05*t), 0.1*np.sin(0.1*t), 0.])
            return omega_b, a_n
        # spiral
        omega_b = np.array([0., 0., 0.03])
        r = 0.5 + 0.01*t
        a_n = np.array([r*np.cos(0.05*t), r*np.sin(0.05*t), 0.01])
        return omega_b, a_n

# ───────────────────── Sensor simulator ────────────────────────────────

@dataclass
class SensorBias:
    gyro: np.ndarray
    acc: np.ndarray

class SensorSimulator:
    def __init__(self):
        self.bias = SensorBias(gyro=np.array([0.01, -0.02, 0.015]),
                               acc=np.array([0.10, -0.10, 0.05]))

    def imu_measurement(self, omega_b: np.ndarray, a_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gyro_m = omega_b + self.bias.gyro + CFG.sigma_gyro*np.random.randn(3)
        acc_m  = a_b    + self.bias.acc  + CFG.sigma_acc *np.random.randn(3)
        return gyro_m, acc_m

    def gps_measurement(self, p_true: np.ndarray) -> np.ndarray:
        return p_true + CFG.sigma_gps*np.random.randn(3)

# ───────────────────── Invariant ESKF class ────────────────────────────

class InvariantESKF:
    """Right-invariant ESKF on SE₂(3) (+ biases, scale)."""

    def __init__(self):
        # nominal state
        self.R = np.eye(3)
        self.v = np.zeros(3)
        self.p = np.zeros(3)
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.s  = 1.0 if CFG.use_scale else 0.0
        # covariances
        self.P  = np.eye(STATE_SIZE)*1e-3
        self.Qd = self._make_Qd()
        self.R_gps = np.eye(3)*CFG.sigma_gps**2

    # ───── public API ────────────────────────────────────────────────
    def predict(self, gyro_m: np.ndarray, acc_m: np.ndarray):
        self._propagate_nominal_rk2(gyro_m, acc_m)
        self._propagate_covariance(gyro_m, acc_m)

    def update_gps(self, z: np.ndarray):
        # innovation h(x) = p + R*lever
        lever_n = self.R @ CFG.lever_b if CFG.use_lever else 0.0
        y = z - (self.p + lever_n)
        H = np.zeros((3, STATE_SIZE))
        # columns: δθ,δv,δp,…  (δp block is columns 6-8)
        H[:,6:9] = self.R
        if CFG.use_lever:
            H[:,0:3] = -self.R @ skew(CFG.lever_b)
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = (K @ y).flatten()
        self._inject(dx)
        I_KH = np.eye(STATE_SIZE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_gps @ K.T

    # ───── internal helpers ─────────────────────────────────────────
    def _make_Qd(self):
        dt = CFG.dt
        Qc = np.zeros((STATE_SIZE, STATE_SIZE))
        Qc[0:3,0:3]   = CFG.sigma_gyro**2*np.eye(3)
        Qc[3:6,3:6]   = CFG.sigma_acc **2*np.eye(3)
        Qc[9:12,9:12] = CFG.sigma_bg_rw**2*np.eye(3)
        Qc[12:15,12:15] = CFG.sigma_ba_rw**2*np.eye(3)
        if CFG.use_scale:
            Qc[15,15] = CFG.sigma_scale_rw**2
        return Qc*dt

    # Nominal state propagation (Euler)
    def _propagate_nominal_rk2(self,
                               gyro_m: np.ndarray,
                               acc_m:  np.ndarray):
        """Runge-Kutta 2（mid-point）积分器，Right-Invariant 形式."""
        dt   = CFG.dt
        omega = gyro_m - self.bg
        acc_b = (acc_m - self.ba) * (self.s if CFG.use_scale else 1.0)

        # ---------- step-1：在 t_k 处求导 ----------
        R_k  = self.R
        a_n1 = R_k @ acc_b + CFG.g_n
        if CFG.use_earth:
            a_n1 += 2 * np.cross([0, 0, CFG.omega_ie], self.v)

        # ---------- step-2：到 t_k+½dt 的中点 ----------
        R_half = R_k @ exp_so3(omega * 0.5 * dt)
        v_half = self.v + 0.5 * a_n1 * dt
        p_half = self.p + 0.5 * self.v * dt

        # 再次评估加速度
        a_n2 = R_half @ acc_b + CFG.g_n
        if CFG.use_earth:
            a_n2 += 2 * np.cross([0, 0, CFG.omega_ie], v_half)

        # ---------- step-3：完成积分 ----------
        self.R = R_k @ exp_so3(omega * dt)         # SO(3) 对数映射
        self.v += a_n2 * dt                        # v_{k+1}
        self.p += v_half * dt                      # p_{k+1}

        # 保证旋转矩阵正交
        U, _, Vt = np.linalg.svd(self.R)
        self.R = U @ Vt

    def _propagate_covariance(self, gyro_m: np.ndarray, acc_m: np.ndarray):
        dt = CFG.dt
        omega = gyro_m - self.bg
        acc_b = (acc_m - self.ba)*(self.s if CFG.use_scale else 1.0)
        A = np.zeros((STATE_SIZE, STATE_SIZE))
        # δθ block
        A[0:3,0:3] = -skew(omega)
        # δv dynamics
        A[3:6,0:3] = -skew(acc_b)
        A[3:6,3:6] = -skew(omega)
        # δp dynamics
        A[6:9,3:6] = np.eye(3)
        A[6:9,0:3] = np.zeros((3,3))
        A[6:9,6:9] = -skew(omega)
        # bias dynamics
        A[0:3,9:12] = -np.eye(3)
        A[3:6,12:15] = -self.R
        A[9:12,9:12] = -np.eye(3)/CFG.tau_bg
        A[12:15,12:15] = -np.eye(3)/CFG.tau_ba
        if CFG.use_scale:
            A[3:6,15] = self.R @ acc_b
        Phi = expm(A*dt)
        self.P = Phi @ self.P @ Phi.T + self.Qd

    def _inject(self, dx: np.ndarray):
        δθ = dx[0:3]
        δv = dx[3:6]
        δp = dx[6:9]
        Jr = right_jacobian_SO3(δθ)
        # update nominal
        self.R = self.R @ exp_so3(δθ)
        self.v += self.R @ δv
        self.p += self.R @ δp
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        if CFG.use_scale:
            self.s += dx[15]
        # re-orthonormalise
        U,_,Vt = np.linalg.svd(self.R)
        self.R = U @ Vt
        # remap P : Γ = blkdiag(Jr, Jr, Jr, I ...)
        Γ = np.eye(STATE_SIZE)
        Γ[0:3,0:3] = Jr
        Γ[3:6,3:6] = Jr
        Γ[6:9,6:9] = Jr
        self.P = Γ @ self.P @ Γ.T

# ───────────────────── Visualisation helpers ──────────────────────────

class Visualiser:
    def __init__(self):
        self.fig = plt.figure(figsize=(10,5))
        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax_err = self.fig.add_subplot(222)
        self.ax_vel = self.fig.add_subplot(224)

    def plot_paths(self, p_true: np.ndarray, p_est: np.ndarray, gps: np.ndarray):
        self.ax3d.plot(*p_true.T,'b',label='Truth')
        self.ax3d.plot(*p_est.T,'r',label='IEKF')
        if gps.size: self.ax3d.scatter(*gps.T,c='g',s=6,label='GPS')
        self.ax3d.set_xlabel('X'); self.ax3d.set_ylabel('Y'); self.ax3d.set_zlabel('Z')
        self.ax3d.legend(); self.ax3d.set_title('3-D Trajectory')

    def plot_errors(self, p_true: np.ndarray, p_est: np.ndarray):
        err = norm(p_true-p_est,axis=1)
        t = np.arange(err.size)*CFG.dt
        self.ax_err.plot(t,err,'k'); self.ax_err.set_xlabel('t [s]'); self.ax_err.set_ylabel('Position error [m]')
        self.ax_err.set_title('RMSE'); self.ax_err.grid(True)

    def plot_velocity(self, v_true: np.ndarray, v_est: np.ndarray):
        t = np.arange(v_true.shape[0])*CFG.dt
        self.ax_vel.plot(t,v_true[:,0],'b',label='vx true')
        self.ax_vel.plot(t,v_est[:,0],'r',label='vx est')
        self.ax_vel.grid(True); self.ax_vel.legend(); self.ax_vel.set_xlabel('t [s]'); self.ax_vel.set_ylabel('m/s')

    def show(self):
        plt.tight_layout(); plt.show()

# ───────────────────── simulation loop ────────────────────────────────

def run_simulation(scen: ScenarioType):
    world = Scenario(scen)
    sim = SensorSimulator()
    iekf = InvariantESKF()
    viz = Visualiser()

    steps = int(CFG.duration/CFG.dt)+1
    p_t = np.zeros(3); v_t = np.zeros(3); R_t = np.eye(3)
    true_p = np.zeros((steps,3)); est_p = np.zeros((steps,3))
    true_v = np.zeros((steps,3)); est_v = np.zeros((steps,3))
    gps_log: List[np.ndarray] = []
    gps_interval = int(CFG.gps_period/CFG.dt)

    for k in range(steps):
        t = k*CFG.dt
        ω_b, a_n = world.true_motion(t)
        a_b_true = R_t.T @ (a_n - CFG.g_n)
        # propagate true state
        R_t = R_t @ exp_so3(ω_b*CFG.dt)
        if CFG.use_earth:
            a_n += 2*np.cross([0,0,CFG.omega_ie], v_t)
        v_t += a_n*CFG.dt
        p_t += v_t*CFG.dt + 0.5*a_n*CFG.dt**2
        # IMU and IEKF predict
        gyro_m, acc_m = sim.imu_measurement(ω_b, a_b_true)
        iekf.predict(gyro_m, acc_m)
        # GPS update
        if k % gps_interval==0:
            z = sim.gps_measurement(p_t)
            iekf.update_gps(z)
            gps_log.append(z)
        # log
        true_p[k] = p_t; est_p[k] = iekf.p
        true_v[k] = v_t; est_v[k] = iekf.v

    gps_arr = np.vstack(gps_log) if gps_log else np.zeros((0,3))
    viz.plot_paths(true_p, est_p, gps_arr)
    viz.plot_errors(true_p, est_p)
    viz.plot_velocity(true_v, est_v)
    viz.show()

# ───────────────────── CLI ────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Right-Invariant ESKF demo')
    parser.add_argument('--scenario', choices=['circle','eight','spiral'], default='spiral')
    parser.add_argument('--duration', type=float, default=60.0)
    args = parser.parse_args()
    CFG.duration = args.duration
    mapping = {'circle':ScenarioType.CIRCLE,
               'eight':ScenarioType.FIGURE_EIGHT,
               'spiral':ScenarioType.SPIRAL}
    run_simulation(mapping[args.scenario])