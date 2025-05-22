"""
One-Dimensional Kalman / Extended-Kalman Filter
with optional velocity-scale correction

------------------------------------------------------------------
MODEL_KEY      = "CV"  | "ACC" | "VEL"
ENABLE_SCALE   = True  → 状态变为 [x, v, s]'
                 False →      [x, v]'
------------------------------------------------------------------
CV   : 匀速模型（无控制，无速度观测）
ACC  : 加速度控制，仍只测位置
VEL  : 同 CV，但观测 = 位置 + 速度
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# ========== 用户可调参数 ==========
MODEL_KEY    = "VEL"      # "CV" | "ACC" | "VEL"
ENABLE_SCALE = True       # 是否在状态中加入速度刻度因子 s
DT           = 0.1        # 步长  [s]
SIM_TIME     = 20.0       # 仿真  [s]

GPS_STD      = 0.5        # 位置观测噪声 σ
VEL_STD      = 0.2        # 速度观测噪声 σ (VEL 模式)
ACC_STD      = 0.3        # 加速度过程噪声 σ (影响 Q)

TRUE_V0      = 1.0        # 真值初速度
TRUE_A       = 0.1        # 真值恒加速度 (ACC 模式)
TRUE_SCALE   = 0.9        # 真值速度刻度 (ENABLE_SCALE=True 才用)

np.random.seed(0)         # 结果可复现
show_animation = True
# ==================================


# ---------- 模型矩阵 ----------
def get_matrices():
    """返回 F, B, H, R, dim_x"""
    dim_x = 3 if ENABLE_SCALE else 2

    # ───── F, B ─────
    F = np.eye(dim_x)
    F[0, 1] = DT                      # x += v*DT
    B = np.zeros((dim_x, 1))
    if MODEL_KEY == "ACC":            # 只有 ACC 用到控制输入
        B[0, 0] = 0.5 * DT**2
        B[1, 0] = DT

    # ───── H, R ─────
    if MODEL_KEY == "VEL":            # 同时测位置 + 速度
        if ENABLE_SCALE:
            H = np.array([[1, 0, 0],
                          [0, 1, 0]])
        else:
            H = np.array([[1, 0],
                          [0, 1]])
        R = np.diag([GPS_STD**2, VEL_STD**2])
    else:                             # 仅测位置
        H = np.array([[1, 0, 0]]) if ENABLE_SCALE else np.array([[1, 0]])
        R = np.array([[GPS_STD**2]])

    return F, B, H, R, dim_x


F, B, H, R, DIM_X = get_matrices()

# ---------- 过程噪声 Q ----------
# 采用简单对角形式，维度自动匹配
Q = np.diag([0.5*DT**2 * ACC_STD,        # 位置过程噪声
             DT * ACC_STD,               # 速度过程噪声
             1e-4] if ENABLE_SCALE else  # s 的过程噪声很小
            [0.5*DT**2 * ACC_STD,
             DT * ACC_STD]) ** 2


# ---------- 真值动力学（含 s） ----------
def true_motion(x, u):
    x = x.copy()
    v_eff = x[1, 0] * (x[2, 0] if ENABLE_SCALE else 1.0)
    x[0, 0] += v_eff * DT + 0.5 * DT**2 * u[0, 0]
    x[1, 0] += DT * u[0, 0]
    # s 不变
    return x


# ---------- 非线性预测 + Jacobian ----------
def nl_motion(x, u):
    x_new = x.copy()
    v_eff = x[1, 0] * x[2, 0]
    x_new[0, 0] += v_eff * DT + 0.5 * DT**2 * u[0, 0]
    x_new[1, 0] += DT * u[0, 0]
    return x_new


def jacob_f(x, u):
    s = x[2, 0]
    v = x[1, 0]
    jF = np.eye(3)
    jF[0, 1] = DT * s
    jF[0, 2] = DT * v
    return jF


# ---------- KF / EKF ----------
def predict(x, P, u):
    if ENABLE_SCALE:
        x_pred = nl_motion(x, u)
        jF = jacob_f(x, u)
        P_pred = jF @ P @ jF.T + Q
    else:
        x_pred = F @ x + B @ u
        P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def update(x_pred, P_pred, z):
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K @ y
    P = (np.eye(DIM_X) - K @ H) @ P_pred
    return x, P


# ---------- 初始量 ----------
x_true = np.zeros((DIM_X, 1))
x_true[1, 0] = TRUE_V0
if ENABLE_SCALE:
    x_true[2, 0] = TRUE_SCALE

x_est = np.zeros((DIM_X, 1))
if ENABLE_SCALE:
    x_est[2, 0] = 1.0                 # 初始刻度设 1
P_est = np.eye(DIM_X)

# ---------- 记录 ----------
ts, tr_pos, est_pos, meas_pos = [], [], [], []
scales_true, scales_est = [], []

# ---------- 主循环 ----------
time = 0.0
while time <= SIM_TIME:
    # ----- 真值与测量 -----
    u_true = np.array([[TRUE_A]]) if MODEL_KEY == "ACC" else np.zeros((1, 1))
    x_true = true_motion(x_true, u_true)

    # 观测
    if MODEL_KEY == "VEL":
        z = H @ x_true + np.vstack((
            GPS_STD * np.random.randn(1, 1),
            VEL_STD * np.random.randn(1, 1)))
    else:
        z = H @ x_true + GPS_STD * np.random.randn(1, 1)

    # ----- EKF / KF -----
    x_pred, P_pred = predict(x_est, P_est, u_true)
    x_est,  P_est  = update(x_pred, P_pred, z)

    # ----- 存储 -----
    ts.append(time)
    tr_pos.append(x_true[0, 0])
    est_pos.append(x_est[0, 0])
    meas_pos.append(z[0, 0])
    if ENABLE_SCALE:
        scales_true.append(x_true[2, 0])
        scales_est.append(x_est[2, 0])

    time += DT


# ---------- 绘图 ----------
plt.figure(figsize=(9, 4))
plt.plot(ts, tr_pos,  "-b", label="true pos")
plt.plot(ts, est_pos, "-r", label="est  pos")
plt.scatter(ts, meas_pos, c="g", s=12, alpha=.4, label="meas pos")
plt.title(f"{MODEL_KEY}  |  Scale={'ON' if ENABLE_SCALE else 'OFF'}")
plt.xlabel("time [s]"); plt.ylabel("position")
plt.grid(); plt.legend()

if ENABLE_SCALE:
    plt.figure(figsize=(7,3))
    plt.plot(ts, scales_true, "-b", label="true s")
    plt.plot(ts, scales_est,  "-r", label="est  s")
    plt.ylabel("scale"); plt.xlabel("time [s]")
    plt.title("Velocity scale factor"); plt.grid(); plt.legend()

plt.tight_layout(); plt.show()
