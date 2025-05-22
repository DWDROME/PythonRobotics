#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

DT        = 0.01        # s
SIM_TIME  = 60.0        # s
g_n       = np.array([0, 0, -9.81])

# --- 噪声与协方差 -------------------------------------------------
SIG_GYRO = 0.002        # rad/s  白噪
SIG_ACC  = 0.05         # m/s^2
SIG_BG_RW= 1e-5         # rad/s/√s
SIG_BA_RW= 5e-5         # m/s^2/√s
SIG_GPS  = 0.5          # m



Q_diag = ([0.0]*3 +                # δp   15×15 误差过程噪声
          [SIG_ACC**2]*3 +         # δv  acc 感测噪声映射到 vel
          [SIG_GYRO**2]*3 +        # δθ  gyro 噪声映射到 att
          [SIG_BG_RW**2]*3 +       # δb_g  陀螺零偏随机游走
          [SIG_BA_RW**2]*3)        # δb_a  加计零偏随机游走
Q_d = np.diag(Q_diag) * DT         # 15×15  离散化


R_gps = np.eye(3)*SIG_GPS**2

# --- 工具函数 ----------------------------------------------------
def skew(v):
    x,y,z = v
    return np.array([[0,-z, y],
                     [z, 0,-x],
                     [-y,x, 0]])

def quat_mul(q1, q2):
    r = R.from_quat(q1) * R.from_quat(q2)
    return r.as_quat()

def small_angle_quat(dtheta):
    """δθ(3) -> 小角度四元数 q=[x,y,z,w]"""
    half = 0.5*dtheta
    w = 1 - 0.5*np.dot(half, half)
    return np.hstack((half, w))

def quat_to_rot(q):
    return R.from_quat(q).as_matrix()

# --- 真值轨迹（简单螺旋） ----------------------------------------
def true_dynamics(t):
    """返回 ω_b, a_b (IMU 理想值, body 坐标)"""
    omega_b = np.array([0.0, 0.0, 0.02])             # yaw 匀速
    a_n = np.array([0.1*np.cos(0.1*t), 0.1*np.sin(0.1*t), 0.0])  # 水平圆周向心
    return omega_b, a_n

# --- EKF 类 ------------------------------------------------------
class ESKF16:
    def __init__(self):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array([0,0,0,1])   # w 最后
        self.bg= np.zeros(3)
        self.ba= np.zeros(3)
        self.P = np.eye(15)*1e-3

    # -------- 预测 --------
    def predict(self, omega_m, acc_m):
        # 去偏
        omega = omega_m - self.bg
        acc_b = acc_m  - self.ba

        R_nb = quat_to_rot(self.q)
        acc_n = R_nb @ acc_b + g_n

        # 主态积分
        self.p += self.v*DT + 0.5*acc_n*DT**2
        self.v += acc_n*DT
        dq     = small_angle_quat(omega*DT)
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q)

        # 误差状态 Jacobian (15x15)
        F = np.zeros((15,15))
        F[0:3,3:6] = np.eye(3)
        F[3:6,6:9] = -R_nb @ skew(acc_b)
        F[3:6,12:15] = -R_nb
        F[6:9,6:9]  = -skew(omega)
        F[6:9,9:12] = -np.eye(3)

        Phi = np.eye(15) + F*DT      # 一阶
        self.P = Phi @ self.P @ Phi.T + Q_d

    # -------- 更新 --------
    def update_gps(self, z):
        H = np.zeros((3,15)); H[:,0:3] = np.eye(3)
        y = z - self.p
        S = H @ self.P @ H.T + R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        delta_x = (K @ y).flatten()

        # 误差注入
        self.p += delta_x[0:3]
        self.v += delta_x[3:6]
        dq = small_angle_quat(delta_x[6:9])
        self.q = quat_mul(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        self.bg += delta_x[9:12]
        self.ba += delta_x[12:15]

        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_gps @ K.T

# --- 主程序 ------------------------------------------------------
def main():
    np.random.seed(0)
    eskf = ESKF16()

    # 真值 & 零偏 (仅用于仿真)
    p_t = np.zeros(3); v_t=np.zeros(3); q_t=np.array([0,0,0,1])
    bg_t = np.array([0.01, -0.02, 0.015])
    ba_t = np.array([0.1, -0.1 , 0.05])

    log_p_true=[]; log_p_est=[]; log_p_gps=[]

    t = 0.0
    while t <= SIM_TIME:
        # --- 真值积分 ---
        omg_b, a_n_true = true_dynamics(t)
        R_nb = quat_to_rot(q_t)
        a_b   = R_nb.T @ (a_n_true - g_n)       # 逆向
        # 积分
        p_t += v_t*DT + 0.5*a_n_true*DT**2
        v_t += a_n_true*DT
        q_t  = quat_mul(q_t, small_angle_quat(omg_b*DT))
        q_t /= np.linalg.norm(q_t)

        # --- 生成 IMU 原始量 + 白噪 ---
        gyro_m = omg_b + bg_t + SIG_GYRO*np.random.randn(3)
        acc_m  = a_b  + ba_t + SIG_ACC *np.random.randn(3)

        # --- EKF 预测 ---
        eskf.predict(gyro_m, acc_m)

        # --- GPS 每 0.2 s 更新一次 ---
        if int(t/DT)%20 == 0:
            z_gps = p_t + SIG_GPS*np.random.randn(3)
            eskf.update_gps(z_gps)
            log_p_gps.append(z_gps)

        # 日志
        log_p_true.append(p_t.copy())
        log_p_est .append(eskf.p.copy())
        t += DT

    log_p_true = np.array(log_p_true)
    log_p_est  = np.array(log_p_est)
    log_p_gps  = np.array(log_p_gps)

    # --- 绘图 ---
    fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
    ax.plot(*log_p_true.T, 'b', label='True')
    ax.plot(*log_p_est.T , 'r', label='ESKF')
    ax.scatter(*log_p_gps.T, c='g', s=5, label='GPS')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.legend(); ax.set_title('16-state INS/GNSS ESKF Demo')
    plt.show()

if __name__ == "__main__":
    main()
