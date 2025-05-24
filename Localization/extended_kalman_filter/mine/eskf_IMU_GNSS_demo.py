
# -*- coding: utf-8 -*-
"""
16-State INS/GNSS Error-State Kalman Filter Demo 

state  : p(3) v(3) q(4) bg(3) ba(3)
control: ω̃(3) ã(3)
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm

# ──────────────────── 配置 ──────────────────── #

from dataclasses import dataclass, field
import numpy as np
@dataclass
class Cfg:
    DT: float = 0.01
    SIM_TIME: float = 60.0
    g: np.ndarray = field(default_factory=lambda: np.array([0., 0., -9.81]))
    sigma_gyro: float = 2e-3      # rad/s
    sigma_acc:  float = 0.05      # m/s²
    sigma_bg_rw: float = 1e-5     # rad/s/√s
    sigma_ba_rw: float = 5e-5     # m/s²/√s
    sigma_gps:  float = 0.5       # m
    tau_bg: float = 3600.0        # gyro-bias τ
    tau_ba: float = 3600.0        # acc-bias  τ

cfg = Cfg()
# ────────────────────────────────────────────── #

# ---------- 工具函数 ----------
def skew(v):
    x, y, z = v
    return np.array([[0, -z,  y],
                     [z,  0, -x],
                     [-y, x,  0]])

def quat_mul(q1, q2):
    return (R.from_quat(q1) * R.from_quat(q2)).as_quat()

def small_angle_quat(dtheta):
    """δθ(3) → 四元数(x,y,z,w) 一阶近似"""
    half = 0.5 * dtheta
    w = 1.0 - 0.5 * np.dot(half, half)
    return np.hstack((half, w))

def quat_to_rot(q):
    return R.from_quat(q).as_matrix()

# ---------- 真值轨迹 ----------
def true_dynamics(t):
    """输出 IMU 理想角速度、导航系加速度"""
    omega_b = np.array([0.0, 0.0, 0.02])                       # yaw 匀速
    a_n = np.array([0.1*np.cos(0.1*t), 0.1*np.sin(0.1*t), 0.0])  # 圆周向心
    return omega_b, a_n

# ---------- 误差过程噪声 Qd (15×15) ----------
def make_Qd(dt: float):
    # δp 由 acc 噪声双积分得到：σ²·dt⁴/4
    sig_p = (cfg.sigma_acc * dt**2 / 2)**2
    Q_diag = ([sig_p]*3 +
              [cfg.sigma_acc**2]*3 +
              [cfg.sigma_gyro**2]*3 +
              [cfg.sigma_bg_rw**2]*3 +
              [cfg.sigma_ba_rw**2]*3)
    return np.diag(Q_diag) * dt
Q_d = make_Qd(cfg.DT)
R_gps = np.eye(3) * cfg.sigma_gps**2
CHI2_THRESH = 16.27   # χ²(3, 0.999)  GPS 异常判阈

# ──────────────────── ESKF 主体 ──────────────────── #
class ESKF16:
    def __init__(self):
        self.p  = np.zeros(3)
        self.v  = np.zeros(3)
        self.q  = np.array([0,0,0,1])  # 四元数 w 分量在末
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.P  = np.eye(15) * 1e-3

    # —— 预测 —— #
    def predict(self, omega_m, acc_m):
        dt = cfg.DT
        g  = cfg.g

        # 1) Nominal 积分
        omega = omega_m - self.bg
        acc_b = acc_m  - self.ba
        R_nb  = quat_to_rot(self.q)
        acc_n = R_nb @ acc_b + g

        self.p += self.v*dt + 0.5*acc_n*dt**2
        self.v += acc_n*dt
        self.q  = quat_mul(self.q, small_angle_quat(omega*dt))
        self.q /= np.linalg.norm(self.q)

        # 2) 误差 F
        F = np.zeros((15,15))
        F[0:3,3:6]  = np.eye(3)
        F[3:6,6:9]  = -R_nb @ skew(acc_b)
        F[3:6,12:15]= -R_nb
        F[6:9,6:9]  = -skew(omega)
        F[6:9,9:12] = -np.eye(3)
        F[9:12,9:12]= -np.eye(3)/cfg.tau_bg
        F[12:15,12:15]= -np.eye(3)/cfg.tau_ba

        # 3) 严格离散化 Φ = exp(FΔt)
        Phi = expm(F*dt)
        self.P = Phi @ self.P @ Phi.T + Q_d

    # —— 更新 —— #
    def update_gps(self, z):
        H = np.zeros((3,15)); H[:,0:3] = np.eye(3)
        y = z - self.p
        S = H @ self.P @ H.T + R_gps

        # Mahalanobis 监测
        if y.T @ np.linalg.solve(S, y) > CHI2_THRESH:
            return  # 丢弃野值

        K = self.P @ H.T @ np.linalg.inv(S)
        delta = (K @ y).flatten()

        # 误差注入
        self.p  += delta[0:3]
        self.v  += delta[3:6]
        self.q   = quat_mul(self.q, small_angle_quat(delta[6:9]))
        self.q  /= np.linalg.norm(self.q)
        self.bg += delta[9:12]
        self.ba += delta[12:15]

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gps @ K.T

# ──────────────────── 主程序 ──────────────────── #
def main():
    eskf = ESKF16()

    # 真值 & (固定) bias
    p_t = np.zeros(3); v_t = np.zeros(3); q_t = np.array([0,0,0,1])
    bg_t = np.array([0.01,-0.02,0.015])
    ba_t = np.array([0.10,-0.10,0.05])

    N  = int(cfg.SIM_TIME / cfg.DT) + 1
    log_true = np.zeros((N,3))
    log_est  = np.zeros((N,3))
    log_gps  = []

    for k in range(N):
        t = k*cfg.DT
        # ---- 生成真值 & IMU ----
        omg_b, a_n_true = true_dynamics(t)
        R_nb = quat_to_rot(q_t)
        a_b  = R_nb.T @ (a_n_true - cfg.g)

        p_t += v_t*cfg.DT + 0.5*a_n_true*cfg.DT**2
        v_t += a_n_true*cfg.DT
        q_t  = quat_mul(q_t, small_angle_quat(omg_b*cfg.DT))
        q_t /= np.linalg.norm(q_t)

        gyro_m = omg_b + bg_t + cfg.sigma_gyro*np.random.randn(3)
        acc_m  = a_b  + ba_t + cfg.sigma_acc *np.random.randn(3)

        # ---- ESKF ----
        eskf.predict(gyro_m, acc_m)
        if k % int(0.2/cfg.DT) == 0:      # 5 Hz GPS
            z = p_t + cfg.sigma_gps*np.random.randn(3)
            eskf.update_gps(z)
            log_gps.append(z)

        log_true[k] = p_t
        log_est [k] = eskf.p

    # ---- 绘图 ----
    log_gps = np.array(log_gps)
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.plot(*log_true.T, 'b', label='True')
    ax.plot(*log_est.T , 'r', label='ESKF')
    ax.scatter(*log_gps.T, c='g', s=6, label='GPS')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); ax.set_title('INS/GNSS 16-State ESKF Demo')
    plt.show()

if __name__ == "__main__":
    main()
