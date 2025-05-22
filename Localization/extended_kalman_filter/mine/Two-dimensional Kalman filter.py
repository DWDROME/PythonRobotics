"""
2-D Extended Kalman Filter  ×  Velocity-Scale Correction (可选)

状态    :
  ENABLE_SCALE=False → [x, y, vx, vy]'
  ENABLE_SCALE=True  → [x, y, vx, vy, s]'   (v_eff = v * s)

MODEL_KEY:
  "CV"  匀速，观测仅 GPS 位置
  "ACC" 加速度控制，观测仅 GPS 位置
  "VEL" 匀速 + 速度观测（GPS 位置 + 轮编/IMU速度）

"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────── 配置区 ────────────────────── #
MODEL_KEY    = "ACC"     # "CV" | "ACC" | "VEL"
ENABLE_SCALE = False      # 是否估计速度刻度因子 s

DT, SIM_TIME = 0.1, 20.0
GPS_STD      = 0.5       # 位置观测噪声 σ
VEL_STD      = 0.2       # 速度观测噪声 σ (仅 VEL)
ACC_STD      = 0.3       # 加速度过程噪声 σ

TRUE_V0      = np.array([[1.0], [0.5]])   # 真值初速度 vx, vy
TRUE_A       = np.array([[0.1], [0.0]])   # ACC 模式恒加速度
TRUE_SCALE   = 0.9                        # 真实刻度 (ENABLE_SCALE 才用)
np.random.seed(0)
# ----------------------------------------------------


# =============== 动力学 / 观测矩阵生成 =================
def get_matrices():
    dim_x = 5 if ENABLE_SCALE else 4
    # ── F, B ──
    F = np.eye(dim_x)
    F[0, 2], F[1, 3] = DT, DT            # 位置积分
    B = np.zeros((dim_x, 2))
    if MODEL_KEY == "ACC":               # 控制=加速度
        B[0, 0] = 0.5*DT**2; B[1, 1] = 0.5*DT**2
        B[2, 0] = DT;        B[3, 1] = DT

    # ── H, R ──
    if MODEL_KEY == "VEL":               # 位置 + 速度观测
        if ENABLE_SCALE:
            H = np.array([[1,0,0,0,0],
                          [0,1,0,0,0],
                          [0,0,1,0,0],
                          [0,0,0,1,0]])
        else:
            H = np.array([[1,0,0,0],
                          [0,1,0,0],
                          [0,0,1,0],
                          [0,0,0,1]])
        R = np.diag([GPS_STD**2, GPS_STD**2,
                     VEL_STD**2, VEL_STD**2])
    else:                               # 仅位置观测
        H = np.array([[1,0,0,0,0],
                      [0,1,0,0,0]]) if ENABLE_SCALE else \
            np.array([[1,0,0,0],
                      [0,1,0,0]])
        R = np.diag([GPS_STD**2, GPS_STD**2])
    return F, B, H, R, dim_x


F, B, H, R, DIM_X = get_matrices()

# 过程噪声 Q —— 位置由双积分得到 0.5 a dt²，速度由 a dt
base = np.diag([0.5*DT**2, 0.5*DT**2, DT, DT])
if ENABLE_SCALE:
    Q = np.block([[base,               np.zeros((4,1))],
                  [np.zeros((1,4)), np.array([[1e-4]])]]) ** 2 * ACC_STD**2
else:
    Q = base ** 2 * ACC_STD**2

# =============== 非线性运动函数 (含刻度) ===============
def f_nl(x, u):
    x_new = x.copy()
    s = x[4,0] if ENABLE_SCALE else 1.0
    # 位置
    x_new[0,0] += (x[2,0]*s)*DT + 0.5*DT**2*u[0,0]
    x_new[1,0] += (x[3,0]*s)*DT + 0.5*DT**2*u[1,0]
    # 速度
    x_new[2,0] += DT*u[0,0]
    x_new[3,0] += DT*u[1,0]
    return x_new

def jacob_f(x, u):
    s = x[4,0]
    jF = np.eye(5)
    jF[0,2] = DT*s; jF[1,3] = DT*s
    jF[0,4] = DT*x[2,0]
    jF[1,4] = DT*x[3,0]
    return jF


# =============== 真值仿真 & 观测生成 ===============
def true_motion(x, u):
    return f_nl(x, u)

def make_measure(x):
    if MODEL_KEY == "VEL":
        noise = np.vstack(( np.random.randn(2,1)*GPS_STD,
                            np.random.randn(2,1)*VEL_STD))
    else:
        noise = np.random.randn(2,1)*GPS_STD
    return H @ x + noise


# =============== KF / EKF 主流程 ===============
def predict(x, P, u):
    if ENABLE_SCALE:
        x_pred = f_nl(x, u)
        F_jac  = jacob_f(x, u)
        P_pred = F_jac @ P @ F_jac.T + Q
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


# =============== 初始化 ===============
x_true = np.zeros((DIM_X,1))
x_true[2:4,0] = TRUE_V0.ravel()
if ENABLE_SCALE: x_true[4,0] = TRUE_SCALE

x_est  = x_true * 0.0
if ENABLE_SCALE: x_est[4,0]  = 1.0          # 估计刻度起始值 1
P_est  = np.eye(DIM_X)

# =============== 仿真循环 ===============
ts, pos_t, pos_e, pos_m, scale_t, scale_e = [], [], [], [], [], []
t = 0.0
while t <= SIM_TIME:
    # 控制 (仅 ACC 用)
    # u = TRUE_A if MODEL_KEY == "ACC" else np.zeros((2,1))
    u = TRUE_A if MODEL_KEY == "ACC" else TRUE_V0.copy()

    # 真实运动
    x_true = true_motion(x_true, u)
    # 观测
    z = make_measure(x_true)
    # 预测 / 更新
    x_pred, P_pred = predict(x_est, P_est, u)
    x_est,  P_est  = update(x_pred, P_pred, z)

    # 记录
    ts.append(t)
    pos_t.append(x_true[:2,0])
    pos_e.append(x_est [:2,0])
    pos_m.append(z[:2,0])
    if ENABLE_SCALE:
        scale_t.append(x_true[4,0]); scale_e.append(x_est[4,0])

    t += DT

# =============== 画图 ===============
pos_t = np.array(pos_t); pos_e = np.array(pos_e); pos_m=np.array(pos_m)
plt.figure(figsize=(6,6))
plt.plot(pos_t[:,0], pos_t[:,1], "-b", label="true")
plt.plot(pos_e[:,0], pos_e[:,1], "-r", label="est")
plt.scatter(pos_m[:,0], pos_m[:,1], c="g", s=10, alpha=.4, label="GPS")
plt.axis("equal"); plt.grid(); plt.legend()
plt.title(f"{MODEL_KEY}  |  Scale={'ON' if ENABLE_SCALE else 'OFF'}")

if ENABLE_SCALE:
    plt.figure(); plt.plot(ts, scale_t, "-b", label="true s")
    plt.plot(ts, scale_e, "-r", label="est  s")
    plt.xlabel("time [s]"); plt.ylabel("scale"); plt.grid(); plt.legend()
    plt.title("Velocity Scale Factor")

plt.show()

