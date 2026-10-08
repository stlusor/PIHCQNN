import torch
import matplotlib.pyplot as plt
from pennylane import numpy as np
import numpy
import pennylane as qml
import matplotlib as mpl
import time
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go

torch.cuda.set_device(0)

num = 40
x_f = torch.unsqueeze(torch.linspace(-1, 1, num).repeat(num), dim=1)
y_f_1 = torch.unsqueeze(torch.linspace(-1, 1, num), dim=1)
y_f = torch.unsqueeze(torch.linspace(0, 0, num ** 2), dim=1)
for i in range(num):
    for j in range(num):
        y_f[num * i + j] = y_f_1[i]

def grt(x, y):
    return 0.5 * (x * y) ** 2

grtruth = grt(x_f, y_f).data.numpy().reshape(num, num).T


mpl.rcParams['font.sans-serif']='Times New Roman'
mpl.rcParams['mathtext.fontset']='stix'
fonten = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\Times.ttf', size=14)
iteration = np.linspace(0, 5000, 100).reshape(-1, 1)
# cut = 300

dim = 40
u_1 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PIHCQNN-case3-2/prediction.txt', header=None).values[:, :]).reshape(dim, dim).T
# u_1 = np.fliplr(u_1)
u_1 = np.matrix(u_1)
layout = go.Layout(width=900, height=900)
fig = go.Figure(data=go.Heatmap(z=u_1, zsmooth='best', colorscale='rainbow', zmax=0.5, zmin=0), layout=layout)
fig.show()

u_1 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PIHCQNN-case3-2/prediction.txt', header=None).values[:, :]).reshape(dim, dim).T
# u_1 = np.fliplr(u_1)
u_1 = np.matrix(u_1)
layout = go.Layout(width=900, height=900)
fig = go.Figure(data=go.Heatmap(z=np.fliplr(u_1 - grtruth), zsmooth='best', colorscale='rainbow', zmax=0.01, zmin=-0.01), layout=layout)
fig.show()

u_1 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PINN-case3-1/prediction.txt', header=None).values[:, :]).reshape(dim, dim).T
# u_1 = np.fliplr(u_1)
u_1 = np.matrix(u_1)
layout = go.Layout(width=900, height=900)
fig = go.Figure(data=go.Heatmap(z=u_1, zsmooth='best', colorscale='rainbow', zmax=0.5, zmin=0), layout=layout)
fig.show()

u_1 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PINN-case3-1/prediction.txt', header=None).values[:, :]).reshape(dim, dim).T
# u_1 = np.fliplr(u_1)
u_1 = np.matrix(u_1)
layout = go.Layout(width=900, height=900)
fig = go.Figure(data=go.Heatmap(z=np.fliplr(u_1 - grtruth), zsmooth='best', colorscale='rainbow', zmax=0.01, zmin=-0.01), layout=layout)
fig.show()

u_1 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PINN-case3-1-less/prediction.txt', header=None).values[:, :]).reshape(dim, dim).T
# u_1 = np.fliplr(u_1)
u_1 = np.matrix(u_1)
layout = go.Layout(width=900, height=900)
fig = go.Figure(data=go.Heatmap(z=u_1, zsmooth='best', colorscale='rainbow', zmax=0.5, zmin=0), layout=layout)
fig.show()

u_1 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PINN-case3-1-less/prediction.txt', header=None).values[:, :]).reshape(dim, dim).T
# u_1 = np.fliplr(u_1)
u_1 = np.matrix(u_1)
layout = go.Layout(width=900, height=900)
fig = go.Figure(data=go.Heatmap(z=np.fliplr(u_1 - grtruth), zsmooth='best', colorscale='rainbow', zmax=0.01, zmin=-0.01), layout=layout)
fig.show()

loss_pihcqnn = np.array(pd.read_csv('C:/Users/REZ/Desktop/PIHCQNN-case3-1/loss.txt', header=None).values[:, 0])
loss_pinn_4 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PINN-case3-1-less/loss.txt', header=None).values[:, 0])
loss_pinn_5 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PINN-case3-2/loss.txt', header=None).values[:, 0])

plt.figure(figsize=(4.5,3.5))
plt.plot(iteration, loss_pihcqnn, label='PIHCQNN')
plt.plot(iteration, loss_pinn_4, label='5-layer PINN')
plt.plot(iteration, loss_pinn_5, label='6-layer PINN')
plt.title('Total Loss', fontproperties=fonten)
plt.ylabel('Value', fontproperties=fonten)
plt.xlabel('Iteration', fontproperties=fonten)
plt.yscale('log', base=10)
plt.xticks(fontproperties=fonten)
plt.yticks(fontproperties=fonten)
plt.legend(prop=fonten)
plt.tight_layout()
plt.show()

loss_pihcqnn = np.array(pd.read_csv('C:/Users/REZ/Desktop/PIHCQNN-case3-1/l2error.txt', header=None).values[:, 0])
loss_pinn_4 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PINN-case3-1-less/l2error.txt', header=None).values[:, 0])
loss_pinn_5 = np.array(pd.read_csv('C:/Users/REZ/Desktop/PINN-case3-2/l2error.txt', header=None).values[:, 0])

plt.figure(figsize=(4.5,3.5))
plt.plot(iteration, loss_pihcqnn, label='PIHCQNN')
plt.plot(iteration, loss_pinn_4, label='5-layer PINN')
plt.plot(iteration, loss_pinn_5, label='6-layer PINN')
plt.title('$\it{l_{2}}$ Error', fontproperties=fonten)
plt.ylabel('Value', fontproperties=fonten)
plt.xlabel('Iteration', fontproperties=fonten)
plt.yscale('log', base=10)
plt.xticks(fontproperties=fonten)
plt.yticks(fontproperties=fonten)
plt.ylim(ymax=1.1)
plt.legend(prop=fonten)
plt.tight_layout()
plt.show()
