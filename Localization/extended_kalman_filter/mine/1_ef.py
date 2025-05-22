import numpy as np
import matplotlib.pyplot as plt

# ── KF 超参数 ──────────────────────────────────────────────
DT       = 0.1     # 采样周期(这里只影响仿真 true_pos 曲线，可省)
SIM_TIME = 20.0    # 仿真时长  [s]
Q        = 0.1     # 过程噪声方差
R        = 1.0     # 测量噪声方差
GPS_STD  = np.sqrt(R)  # 便于直观写噪声

# ── 1-D KF 实现 ────────────────────────────────────────────
def kf_1d(zs, x0=0.0, P0=1.0, Q=0.1, R=1.0):
    """
    一维 Kalman 滤波
      zs : 观测序列
      x0, P0 : 初始状态估计及协方差
      Q, R   : 过程 / 测量噪声方差
    返回 x̂[k] 数组
    """
    x, P = x0, P0
    xs   = np.zeros_like(zs)

    for k, z in enumerate(zs):
        # —— 预测 ——
        x_pred = x
        P_pred = P + Q

        # —— 更新 ——
        K   = P_pred / (P_pred + R)
        x   = x_pred + K * (z - x_pred)
        P   = (1 - K) * P_pred
        xs[k] = x

    return xs

# ── 主程序 ────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    steps      = int(SIM_TIME / DT)
    true_pos   = 24.0                               # “真值”保持常数
    measurements = true_pos + np.random.randn(steps) * GPS_STD  # 带噪观测

    estimates = kf_1d(measurements, x0=0.0, P0=1.0, Q=Q, R=R)

    # 绘图
    t = np.arange(steps) * DT
    plt.figure(figsize=(8,4))
    plt.plot(t, measurements, label="measure (z)")
    plt.plot(t, estimates,   label="KF estimate")
    plt.hlines(true_pos, t[0], t[-1], colors="k", linestyles="--", label="true pos")
    plt.xlabel("time [s]")
    plt.ylabel("position")
    plt.title("1-D Kalman Filter")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.show()
