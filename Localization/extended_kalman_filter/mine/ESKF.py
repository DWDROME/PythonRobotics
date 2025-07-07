"""
16维度的INS+GNSS , 一个供学习的小demo
============================================================================

----------
1. **Config-driven** – 调节一些实验项目
2. **Scenario plug-ins** – 选择不同的运动轨迹，如圆形、8字形、或者是自定义轨迹
3. **Modular classes** – `ESKF`, `传感器数据模拟`, `可视化处理`.
4. **Extensible state** – 可选的刻度因子、IMU-GNSS 杠杆臂
5. **Rich plots** – 3D 轨迹、位置误差、速度误差

python ESKF.py --scenario circle --duration 120
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

# ════════════════════════════════════════════════════════════════════════
#  ESKF 工具函数
# ════════════════════════════════════════════════════════════════════════


def skew(v: np.ndarray):
    """Return the 3×3 cross-product matrix of a vector."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def small_quaternion(dtheta: np.ndarray):
    """Convert a small rotation vector to a quaternion (x, y, z, w)."""
    half = 0.5 * dtheta
    w = 1.0 - 0.5 * half.dot(half)
    return np.hstack((half, w))


def quat_mul(q1: np.ndarray, q2: np.ndarray) :
    """Quaternion multiplication (x, y, z, w convention)."""
    return (R.from_quat(q1) * R.from_quat(q2)).as_quat()


def quat_to_rot(q: np.ndarray):
    return R.from_quat(q).as_matrix()


# ════════════════════════════════════════════════════════════════════════
# 1. 全局设置
# ════════════════════════════════════════════════════════════════════════
class ScenarioType(Enum):
    CIRCLE = auto()
    FIGURE_EIGHT = auto()
    CUSTOM = auto()
    SPIRAL = auto()

@dataclass
class GlobalConfig:
    # Simulation timing
    dt: float = 0.01
    duration: float = 60.0
    # Sensors
    sigma_gyro: float = 0.002     # rad/s
    sigma_acc: float = 0.05       # m/s²
    sigma_bg_rw: float = 1e-5     # gyro-bias 随机游走率
    sigma_ba_rw: float = 5e-5     # acc-bias 随机游走率
    sigma_gps: float = 0.5        # m
    # Process model
    tau_bg: float = 3600.0        # gyro-bias correlation time (s)
    tau_ba: float = 3600.0        # acc-bias correlation time (s)
    g_n: np.ndarray = field(default_factory=lambda: np.array([0., 0., -9.81]))  # m/s²
    # Earth params
    use_earth_rotation: bool = True
    omega_ie: float = 7.292115e-5  # rad/s
    # Lever-arm (IMU→GNSS)
    use_lever: bool = True
    lever_b: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.2]))
    # Scale factor
    use_scale: bool = True
    sigma_scale_rw: float = 2e-4
    # Visualization
    gps_period: float = 0.2

CFG = GlobalConfig()

# Calculate state dimension dynamically
STATE_SIZE = 16 if CFG.use_scale else 15


# ════════════════════════════════════════════════════════════════════════
# 2. 运动轨迹
# ════════════════════════════════════════════════════════════════════════
class Scenario:
    """Generate true motion for different test cases."""

    def __init__(self, scenario: ScenarioType):
        self.scenario = scenario

    def true_motion(self, t: float):
        if self.scenario is ScenarioType.CIRCLE:
            omega_b = np.array([0.0, 0.0, 0.02])
            a_n = np.array([0.1 * np.cos(0.05 * t), 0.1 * np.sin(0.05 * t), 0.0])
            return omega_b, a_n
        if self.scenario is ScenarioType.FIGURE_EIGHT:
            omega_b = np.array([0.0, 0.0, 0.04 * np.sin(0.05 * t)])
            a_n = np.array([
                0.2 * np.sin(0.05 * t),
                0.1 * np.sin(0.1 * t),
                0.0,
            ])
            return omega_b, a_n
        
        if self.scenario is ScenarioType.SPIRAL:
            omega_b = np.array([0, 0, 0.03])
            r = 0.5 + 0.01*t
            a_n = np.array([r*np.cos(0.05*t), r*np.sin(0.05*t), 0.01])
            return omega_b, a_n

        # CUSTOM – placeholder for user-defined path
        omega_b = np.zeros(3)
        a_n = np.zeros(3)
        return omega_b, a_n


# ════════════════════════════════════════════════════════════════════════
# 3. 传感器数据模拟
# ════════════════════════════════════════════════════════════════════════
@dataclass
class SensorBiases:
    gyro: np.ndarray
    acc: np.ndarray

class SensorSimulator:
    """Generate noisy IMU and GPS measurements given true motion."""

    def __init__(self):
        self.bias = SensorBiases(
            gyro=np.array([0.01, -0.02, 0.015]),
            acc=np.array([0.10, -0.10, 0.05]),
        )

    def imu_measurement(self, omega_b: np.ndarray, a_b: np.ndarray):
        gyro_m = omega_b + self.bias.gyro + CFG.sigma_gyro * np.random.randn(3)
        acc_m = a_b + self.bias.acc + CFG.sigma_acc * np.random.randn(3)
        return gyro_m, acc_m

    def gps_measurement(self, p_true: np.ndarray) :
        return p_true + CFG.sigma_gps * np.random.randn(3)


# ════════════════════════════════════════════════════════════════════════
# 4. 扩展卡尔曼滤波器 (ESKF)
# ════════════════════════════════════════════════════════════════════════
class ESKF:
    """16-state ESKF with optional lever-arm and scale-factor."""

    def __init__(self):
        # Nominal state
        self.p = np.zeros(3)    # 位置
        self.v = np.zeros(3)    # 速度
        self.q = np.array([0, 0, 0, 1])  # 姿态四元数(x, y, z, w)
        self.bg = np.zeros(3)   # 陀螺仪偏执
        self.ba = np.zeros(3)   # 加速度计偏执
        self.s = 1.0 if CFG.use_scale else 0.0  # 刻度因子
        # Error covariance
        self.P = np.eye(STATE_SIZE) * 1e-3 # 误差协方差矩阵
        # Pre-compute static noise matrices
        self.Qd = self._make_Qd()   # 过程噪声协方差矩阵
        self.R_gps = np.eye(3) * CFG.sigma_gps ** 2 # GPS 观测噪声协方差矩阵

    # ---- Core public API ----
    def predict(self, gyro_m: np.ndarray, acc_m: np.ndarray):
        self._nominal_rk2(gyro_m, acc_m)
        self._propagate_covariance(gyro_m, acc_m)

    def update_gps(self, z: np.ndarray):
        lever_n = quat_to_rot(self.q) @ CFG.lever_b if CFG.use_lever else 0.0
        h = self.p + lever_n
        y = z - h
        H = np.zeros((3, STATE_SIZE))
        H[:, 0:3] = np.eye(3)
        if CFG.use_lever:
            H[:, 6:9] = -quat_to_rot(self.q) @ skew(CFG.lever_b)
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = (K @ y).flatten()
        self._inject(dx)
        I_KH = np.eye(STATE_SIZE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_gps @ K.T

    # ---- 内部函数 ----
    def _make_Qd(self):
        dt = CFG.dt
        sig_p = (CFG.sigma_acc * dt ** 2 / 2) ** 2
        diag = (
            [sig_p] * 3
            + [CFG.sigma_acc ** 2] * 3
            + [CFG.sigma_gyro ** 2] * 3
            + [CFG.sigma_bg_rw ** 2] * 3
            + [CFG.sigma_ba_rw ** 2] * 3
        )
        if CFG.use_scale:
            diag.append(CFG.sigma_scale_rw ** 2)
        return np.diag(diag) * dt

    def _nominal_rk2(self, gyro_m: np.ndarray, acc_m: np.ndarray):
        dt = CFG.dt
        # 移除偏置和刻度因子
        omega = gyro_m - self.bg
        acc_b = (acc_m - self.ba) * (self.s if CFG.use_scale else 1.0)
        # 第一步预测
        Rnb = quat_to_rot(self.q)
        acc_n1 = Rnb @ acc_b + CFG.g_n
        if CFG.use_earth_rotation:
            acc_n1 += 2 * np.cross([0, 0, CFG.omega_ie], self.v)
        v_half = self.v + 0.5 * acc_n1 * dt
        p_half = self.p + 0.5 * self.v * dt
        q_half = quat_mul(self.q, small_quaternion(omega * dt * 0.5))
        # 第二步预测
        acc_n2 = quat_to_rot(q_half) @ acc_b + CFG.g_n
        if CFG.use_earth_rotation:
            acc_n2 += 2 * np.cross([0, 0, CFG.omega_ie], v_half)
        # 更新状态
        self.v += acc_n2 * dt
        self.p += v_half * dt
        self.q = quat_mul(self.q, small_quaternion(omega * dt))
        self.q /= np.linalg.norm(self.q)

    def _propagate_covariance(self, gyro_m: np.ndarray, acc_m: np.ndarray):
        dt = CFG.dt
        omega = gyro_m - self.bg
        acc_b = (acc_m - self.ba) * (self.s if CFG.use_scale else 1.0)
        Rnb = quat_to_rot(self.q)
        F = np.zeros((STATE_SIZE, STATE_SIZE))
        F[0:3, 3:6] = np.eye(3)
        F[3:6, 6:9] = -Rnb @ skew(acc_b)
        F[3:6, 12:15] = -Rnb
        F[6:9, 6:9] = -skew(omega)
        F[6:9, 9:12] = -np.eye(3)
        F[9:12, 9:12] = -np.eye(3) / CFG.tau_bg
        F[12:15, 12:15] = -np.eye(3) / CFG.tau_ba
        if CFG.use_scale:
            F[3:6, 15] = Rnb @ acc_b
        Phi = expm(F * dt)
        self.P = Phi @ self.P @ Phi.T + self.Qd

    def _inject(self, dx: np.ndarray):
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.q = quat_mul(self.q, small_quaternion(dx[6:9]))
        self.q /= np.linalg.norm(self.q)
        self.bg += dx[9:12]
        self.ba += dx[12:15]
        if CFG.use_scale:
            self.s += dx[15]


# ════════════════════════════════════════════════════════════════════════
# 5. 可视化
# ════════════════════════════════════════════════════════════════════════
class Visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 5))
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax_err = self.fig.add_subplot(222)
        self.ax_vel = self.fig.add_subplot(224)

    def plot_paths(self, true_p: np.ndarray, est_p: np.ndarray, gps_p: np.ndarray):
        self.ax3d.plot(*true_p.T, "b", label="True")
        self.ax3d.plot(*est_p.T, "r", label="ESKF")
        self.ax3d.scatter(*gps_p.T, c="g", s=6, label="GPS")
        self.ax3d.set_xlabel("X")
        self.ax3d.set_ylabel("Y")
        self.ax3d.set_zlabel("Z")
        self.ax3d.legend()
        self.ax3d.set_title("3-D Trajectory")

    def plot_errors(self, true_p: np.ndarray, est_p: np.ndarray):
        pos_err = np.linalg.norm(true_p - est_p, axis=1)
        t = np.arange(pos_err.size) * CFG.dt
        self.ax_err.plot(t, pos_err, "k")
        self.ax_err.set_xlabel("Time [s]")
        self.ax_err.set_ylabel("Position error [m]")
        self.ax_err.set_title("Position RMSE vs Time")
        self.ax_err.grid(True)

    def plot_velocity(self, v_true: np.ndarray, v_est: np.ndarray):
        t = np.arange(v_true.shape[0]) * CFG.dt
        self.ax_vel.plot(t, v_true[:, 0], "b", label="Vx true")
        self.ax_vel.plot(t, v_est[:, 0], "r", label="Vx est")
        self.ax_vel.set_xlabel("Time [s]")
        self.ax_vel.set_ylabel("Velocity [m/s]")
        self.ax_vel.grid(True)
        self.ax_vel.legend()

    def show(self):
        plt.tight_layout()
        plt.show()


# ════════════════════════════════════════════════════════════════════════
# 6. 主程序
# ════════════════════════════════════════════════════════════════════════

def run_simulation(scenario_type: ScenarioType):
    scenario = Scenario(scenario_type)
    imu = SensorSimulator()
    eskf = ESKF()
    viz = Visualizer()

    steps = int(CFG.duration / CFG.dt) + 1
    true_p = np.zeros((steps, 3))
    est_p = np.zeros((steps, 3))
    true_v = np.zeros((steps, 3))
    est_v = np.zeros((steps, 3))
    gps_log: List[np.ndarray] = []
    # 初始状态
    # 位置、速度、姿态四元数
    p_t = np.zeros(3)
    v_t = np.zeros(3)
    q_t = np.array([0, 0, 0, 1])

    gps_interval = int(CFG.gps_period / CFG.dt)

    for k in range(steps):
        t = k * CFG.dt
        # ----- true motion -----
        omega_b, a_n = scenario.true_motion(t)
        Rnb_true = quat_to_rot(q_t)
        a_b_true = Rnb_true.T @ (a_n - CFG.g_n)
        # Propagate true
        p_t += v_t * CFG.dt + 0.5 * a_n * CFG.dt ** 2
        v_t += a_n * CFG.dt
        q_t = quat_mul(q_t, small_quaternion(omega_b * CFG.dt))
        q_t /= np.linalg.norm(q_t)
        # ----- sensors -----
        gyro_m, acc_m = imu.imu_measurement(omega_b, a_b_true)
        eskf.predict(gyro_m, acc_m)
        if k % gps_interval == 0:
            z_gps = imu.gps_measurement(p_t)
            eskf.update_gps(z_gps)
            gps_log.append(z_gps)
        # ----- log -----
        true_p[k] = p_t
        est_p[k] = eskf.p
        true_v[k] = v_t
        est_v[k] = eskf.v

    gps_arr = np.vstack(gps_log)
    viz.plot_paths(true_p, est_p, gps_arr)
    viz.plot_errors(true_p, est_p)
    viz.plot_velocity(true_v, est_v)
    viz.show()


# ════════════════════════════════════════════════════════════════════════
# 7. CLI 入口
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INS/GNSS ESKF demo")
    parser.add_argument(
        "--scenario",
        choices=["circle", "eight","spiral"],
        default="spiral",
        help="Trajectory scenario",
    )
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()
    CFG.duration = args.duration
    mapping = {
        "circle" : ScenarioType.CIRCLE,
        "eight"  : ScenarioType.FIGURE_EIGHT,
        "spiral" : ScenarioType.SPIRAL,
    }
    scen = mapping.get(args.scenario, ScenarioType.CIRCLE)

    # scen = ScenarioType.CIRCLE if args.scenario == "circle" else ScenarioType.FIGURE_EIGHT
    run_simulation(scen)
