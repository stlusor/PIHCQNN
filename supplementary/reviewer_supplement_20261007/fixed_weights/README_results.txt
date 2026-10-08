固定边界权重补充实验（2026-10-07）

本目录包含18组全新CPU实验：PINN/PIHCQNN × λBC=0.1/1/10 × seeds=0/1/2。
统一设置：soft BC；100个等距内部配点；201点评价；冷启动；1000次Adam更新；lr=0.005；CPU float32；每个进程1个PyTorch线程，驱动同时运行最多3个进程。
PIHCQNN：5 qubits, n=1, hidden width=5, 106 trainable parameters（30 quantum + 76 classical）；PINN：hidden width=5, 76 parameters。
L=Lres+λBC Lbc，Lres=mean[(u_xx+4π²sin(2πx))²]，Lbc=u(0)²+u(1)²。
目标精确解 u=sin(2πx)。λBC明确指边界项权重，残差项权重始终为1。

model | λBC | u relative L2 (%) mean ± sample SD | boundary norm mean ± SD | training residual MSE mean ± SD
pinn | 0.1 | 272.9874 ± 29.3368 | 4.06394 ± 1.5997 | 132.823 ± 226.816
pinn | 1 | 228.0925 ± 41.2302 | 3.9629 ± 0.702185 | 7.56012 ± 7.4979
pinn | 10 | 58.7749 ± 100.9242 | 1.09799 ± 1.88733 | 13.3609 ± 23.0482
qpinn | 0.1 | 38.8820 ± 28.1267 | 0.668703 ± 0.48389 | 0.0584325 ± 0.0320601
qpinn | 1 | 7.0317 ± 7.1251 | 0.120982 ± 0.122645 | 0.0382876 ± 0.00130918
qpinn | 10 | 0.2126 ± 0.0601 | 0.00260098 ± 0.00100158 | 0.0467251 ± 0.0156229

boundary norm = sqrt(u(0)^2+u(1)^2)，为绝对端点误差；residual MSE在100个固定训练配点计算。
标准差采用样本SD（ddof=1），所有表值均从本目录全新CPU三seed结果计算。
固定训练预算下权重与初值均影响误差，此矩阵用于固定权重敏感性与训练梯度诊断。
时间记录包含实际训练、检查点诊断及每步有限性检查。

数值核验：六种 model/λBC 组合均与原elastostatic函数公式、梯度和一步Adam更新逐参数一致（最大差值均0）。
18组1000步均完成；18,000次loss检查、18,000次gradient检查与36,000次parameter检查中未记录非有限事件。
每组在状态step=0,100,...,1000记录实际完整PINN损失的量子/经典参数梯度L2与RMS、两项分量梯度及cos角。所有记录与随后一次更新前的参数状态一致。

文件：summary.csv/summary.json为三seed汇总；all_runs.csv为逐seed结果；gradients_history.csv为198条真实训练梯度检查点；runs/*保存训练点、checkpoint、predictions与详细JSON；verification.json及integrity_checks.json保存一致性核验。

运行：
python reviewer_supplement_20261007/fixed_weights/train_fixed_weights.py all --workers 3
python reviewer_supplement_20261007/fixed_weights/audit_results.py
all命令从冷启动重新生成矩阵，覆盖相应run目录。
