"""
3-D Extended Kalman Filter  ×  Velocity-Scale Correction (可选)
---------------------------------------------------------------
状态 (ENABLE_SCALE=False):
    x = [x, y, z, vx, vy, vz]ᵀ
状态 (ENABLE_SCALE=True):
    x = [x, y, z, vx, vy, vz, s]ᵀ     (v_eff = v·s)
控制:
    MODEL_KEY=="CV"  : 无
    MODEL_KEY=="ACC" : u = [ax, ay, az]ᵀ
    MODEL_KEY=="VEL" : u = [vx, vy, vz]ᵀ  (速度观测，而非控制)

"""

import numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D      # noqa: F401  (IDE 提示用)

# ──────────────── 参   数 ──────────────── #
MODEL_KEY      = "VEL"     # "CV" | "ACC" | "VEL"
ENABLE_SCALE   = True      # True→估计速度刻度 s
DT, SIM_TIME   = 0.1, 20.0
GPS_STD, VEL_STD, ACC_STD = 0.3, 0.2, 0.3
TRUE_V0  = np.array([[1.0], [0.5], [0.2]])
TRUE_A   = np.array([[0.1], [0.0], [0.0]])
TRUE_S   = 0.9
np.random.seed(0)
# ───────────────────────────────────────── #

# ------------ 矩阵生成 ------------
def get_linear_FB(dim_x):
    """返回线性 F,B（未考虑刻度）"""
    F = np.eye(dim_x)
    F[0, 3], F[1, 4], F[2, 5] = DT, DT, DT
    B = np.zeros((dim_x, 3))
    if MODEL_KEY == "ACC":
        half = 0.5 * DT**2
        B[:3, :] = np.diag([half, half, half])
        B[3:6, :] = np.diag([DT, DT, DT])
    elif MODEL_KEY == "VEL":
        # VEL 模式没有控制，速度当观测
        pass
    return F, B

def get_H_R():
    """观测矩阵 / 协方差"""
    if MODEL_KEY == "VEL":
        if ENABLE_SCALE:
            H = np.block([
                [np.eye(3), np.zeros((3,3)), np.zeros((3,1))],   # 位置
                [np.zeros((3,3)), np.eye(3), np.zeros((3,1))]   # 速度
            ])
        else:
            H = np.block([
                [np.eye(3), np.zeros((3,3))],
                [np.zeros((3,3)), np.eye(3)]
            ])
        R = np.diag([GPS_STD**2]*3 + [VEL_STD**2]*3)
    else:  # 仅测位置
        H = np.hstack((np.eye(3), np.zeros((3,4)))) if ENABLE_SCALE \
            else np.hstack((np.eye(3), np.zeros((3,3))))
        R = np.diag([GPS_STD**2]*3)
    return H, R

# ---------- 非线性运动（含 s） ----------
def motion_nl(x, u):
    x2 = x.copy()
    s = x[6,0]
    # 位置
    x2[0:3,0] += x[3:6,0]*s*DT + 0.5*DT**2*u[:,0]
    # 速度
    x2[3:6,0] += DT*u[:,0]
    return x2

def jacob_f(x, u):
    """7×7 Jacobian when ENABLE_SCALE=True"""
    s = x[6,0]
    vx, vy, vz = x[3:6,0]
    J = np.eye(7)
    J[0,3] = DT*s;  J[1,4] = DT*s;  J[2,5] = DT*s
    J[0,6] = DT*vx; J[1,6] = DT*vy; J[2,6] = DT*vz
    return J

# ---------- 过程噪声 Q ----------
def build_Q(dim_x):
    base = np.diag([0.5*DT**2]*3 + [DT]*3)
    Q_lin = base @ (ACC_STD**2 * np.eye(6)) @ base.T
    if ENABLE_SCALE:
        Qfull = np.zeros((dim_x, dim_x))
        Qfull[:6,:6] = Q_lin
        Qfull[6,6] = 1e-4
        return Qfull
    return Q_lin

# ---------- 真值运动 & 观测 ----------
def true_motion(x, u):
    return motion_nl(x, u) if ENABLE_SCALE else F_lin @ x + B_lin @ u

def make_measure(x):
    if MODEL_KEY == "VEL":
        noise = np.vstack((np.random.randn(3,1)*GPS_STD,
                           np.random.randn(3,1)*VEL_STD))
    else:
        noise = np.random.randn(3,1)*GPS_STD
    return H @ x + noise

# ========== 初始化 ==========
DIM_X = 7 if ENABLE_SCALE else 6
F_lin, B_lin = get_linear_FB(DIM_X)
H, R = get_H_R()
Q = build_Q(DIM_X)

x_true = np.zeros((DIM_X,1))
x_true[3:6,0] = TRUE_V0.ravel()
if ENABLE_SCALE: x_true[6,0] = TRUE_S

x_est  = np.zeros_like(x_true)
if ENABLE_SCALE: x_est[6,0] = 1.0
P_est  = np.eye(DIM_X)

# ---------- 仿真循环 ----------
ts, pt, pe = [], [], []
scale_t, scale_e = [], []
meas = []
t=0.0

while t <= SIM_TIME:
    u = TRUE_A if MODEL_KEY=="ACC" else np.zeros((3,1))
    # 真实运动
    x_true = true_motion(x_true, u)
    # 观测
    z = make_measure(x_true)
    # 预测
    if ENABLE_SCALE:
        x_pred = motion_nl(x_est, u)
        JF = jacob_f(x_est, u)
        P_pred = JF @ P_est @ JF.T + Q
    else:
        x_pred = F_lin @ x_est + B_lin @ u
        P_pred = F_lin @ P_est @ F_lin.T + Q
    # 更新
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P_est = (np.eye(DIM_X) - K @ H) @ P_pred

    # 记录
    ts.append(t)
    pt.append(x_true[0:3,0]); pe.append(x_est[0:3,0])
    meas.append(z[0:3, 0])          # 只存位置部分 
    if ENABLE_SCALE:
        scale_t.append(x_true[6,0]); scale_e.append(x_est[6,0])
    t += DT

# ---------- 绘图 ----------
pt = np.array(pt); pe = np.array(pe)
meas = np.array(meas)
fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
ax.plot(pt[:,0], pt[:,1], pt[:,2], "-b", label="true")
ax.plot(pe[:,0], pe[:,1], pe[:,2], "-r", label="est")
ax.scatter(meas[:,0], meas[:,1], meas[:,2],
            c="lime", s=20, alpha=.5, label="GPS")
ax.set_title(f"{MODEL_KEY} | Scale={'ON' if ENABLE_SCALE else 'OFF'}")
ax.legend(); ax.grid()

if ENABLE_SCALE:
    plt.figure()
    plt.plot(ts, scale_t, "-b", label="true s")
    plt.plot(ts, scale_e, "-r", label="est  s")
    ax.scatter(meas[:,0], meas[:,1], meas[:,2],
            c="lime", s=20, alpha=0.5, label="GPS")
    plt.title("Velocity Scale Factor")

plt.show()
