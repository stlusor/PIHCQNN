import torch
import matplotlib.pyplot as plt
from pennylane import numpy as np
import numpy
import pennylane as qml
import matplotlib as mpl
import time
import seaborn as sns

torch.cuda.set_device(0)

mpl.rcParams['font.sans-serif']='Times New Roman'
mpl.rcParams['mathtext.fontset']='stix'
fonten = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\Times.ttf', size=14)

### parameter
def source(t, x):
    t_max = 0.5
    sigma = 0.02
    u_max = 1
    p = 0.25 * torch.cos(2 * torch.pi * t / t_max) + 0.5
    p_t = -0.5 * torch.sin(2 * torch.pi * t / t_max) * torch.pi / t_max
    u_sol = u_max * torch.exp(-(x - p) ** 2 / (2 * sigma ** 2))
    k_sol = 0.01 * u_sol + 7
    k_u_sol = 0.01
    c_sol = 0.0005 * u_sol ** 2 + 500
    factor = 1 / (sigma ** 2)
    s = factor * k_sol * u_sol + u_sol * (x - p) * factor * (c_sol * p_t - (x - p) * factor * (k_sol + u_sol * k_u_sol))
    return s

def f(t, x):
    u = model(torch.cat((t, x), dim=1))
    u_t = get_derivative(u, t, 1)
    u_x = get_derivative(u, x, 1)
    u_xx = get_derivative(u, x, 2)
    k = 0.01 * u + 7
    k_u = 0.01
    c = 0.0005 * u ** 2 + 500
    s = source(t, x)
    f = c * u_t - k_u * u_x * u_x - k * u_xx - s
    return f

def u_0(x):
    u0 = torch.exp((-(x - 0.75) ** 2) / (2 * (0.02 ** 2)))
    return u0
def get_derivative(y, x, n):
    if n == 0:
        return y
    else:
        dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True, allow_unused=True)[0]
        return get_derivative(dy_dx, x, n - 1)

n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

n_layers = 2
weight_shapes = {"weights": (n_layers, n_qubits)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

clayer_1 = torch.nn.Linear(2, 10)
clayer_2 = torch.nn.Linear(10, 10)
clayer_comparison = torch.nn.Linear(10, 10)
clayer_3 = torch.nn.Linear(10, 10)
clayer_4 = torch.nn.Linear(10, 1)
# softmax = torch.nn.Softmax(dim=1)
layers = [clayer_1, torch.nn.Tanh(), clayer_2, torch.nn.Tanh(), clayer_comparison, torch.nn.Tanh(), clayer_3, torch.nn.Tanh(), clayer_4]
# layers = [clayer_1, torch.nn.Tanh(), clayer_2, torch.nn.Tanh(), clayer_3, torch.nn.Tanh(), clayer_4]
model = torch.nn.Sequential(*layers)
model = model.cuda()

class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_f, t_b, x_b1, x_b2, t_i, x_i):
        res_f = f(t_f, x_f)
        mse_f_value = torch.mean(torch.pow(res_f, 2))

        pred_b1 = model(torch.cat((t_b, x_b1), dim=1))[:, 0].view(-1, 1)
        pred_b2 = model(torch.cat((t_b, x_b2), dim=1))[:, 0].view(-1, 1)
        pred_b1_ux = get_derivative(pred_b1, x_b1, 1)
        pred_b2_ux = get_derivative(pred_b2, x_b2, 1)
        mse_b_value = torch.mean(torch.pow(pred_b1_ux, 2)) + torch.mean(torch.pow(pred_b2_ux, 2))

        pred_i = model(torch.cat((t_i, x_i), dim=1))[:, 0].view(-1, 1)
        mse_i_value = torch.mean(torch.pow(pred_i - u_0(x_i), 2))
        return 3E-7 * mse_f_value + mse_b_value + mse_i_value

num = 50
x_f = torch.unsqueeze(torch.linspace(0, 1, num).repeat(num), dim=1)
t_f_1 = torch.unsqueeze(torch.linspace(0, 0.5, num), dim=1)
t_f = torch.unsqueeze(torch.linspace(0, 0, num ** 2), dim=1)
for i in range(num):
    for j in range(num):
        t_f[num * i + j] = t_f_1[i]

t_b = torch.unsqueeze(torch.linspace(0, 0.5, 20), dim=1)
x_b1 = torch.unsqueeze(torch.linspace(0, 0, 20), dim=1)
x_b2 = torch.unsqueeze(torch.linspace(1, 1, 20), dim=1)
x_i = torch.unsqueeze(torch.linspace(0, 1, 20), dim=1)
t_i = torch.unsqueeze(torch.linspace(0, 0, 20), dim=1)




def grt(t, x):
    pt = 0.25 * torch.cos(4 * torch.pi * t) + 0.5
    utx = torch.exp((- (x - pt) ** 2) / (2 * (0.02) ** 2))
    return utx

grtruth = grt(t_f, x_f).data.numpy()


x_f.requires_grad = True
t_f.requires_grad = True
t_b.requires_grad = True
x_b1.requires_grad = True
x_b2.requires_grad = True
t_i.requires_grad = True
x_i.requires_grad = True

x_f = x_f.cuda()
t_f = t_f.cuda()
t_b = t_b.cuda()
x_b1 = x_b1.cuda()
x_b2 = x_b2.cuda()
x_i = x_i.cuda()
t_i = t_i.cuda()



optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
loss_function = My_loss()
l2error = []
loss = []
start_time = time.time()
for step in range(50000):
    def closure():
        optimizer.zero_grad()
        loss = loss_function(x_f, t_b, x_b1, x_b2, t_i, x_i)
        # print(loss)
        loss.backward()
        return loss
    optimizer.step(closure)

    if step % 100 == 0:
        loss_tem = loss_function(x_f, t_b, x_b1, x_b2, t_i, x_i)
        loss.append(loss_tem.cpu().data.numpy())
        numpy.savetxt('C:/Users/REZ/Desktop/PINN/loss.txt', loss)
        print(loss_tem)
        prediction = model(torch.cat((t_f, x_f), dim=1))[:, 0].view(-1, 1).cpu().data.numpy()
        numpy.savetxt('C:/Users/REZ/Desktop/PINN/prediction.txt', prediction)
        error = prediction - grtruth
        l2_error = numpy.linalg.norm(error) / numpy.linalg.norm(grtruth)
        print(step, l2_error)
        l2error.append(l2_error)
        numpy.savetxt('C:/Users/REZ/Desktop/PINN/l2error.txt', l2error)

        plt.close()
        sns.heatmap(numpy.flipud(np.array(prediction).reshape(num, num).T), cmap='rainbow')
        plt.title(f'PINN Prediction ({step} Iterations)', fontproperties=fonten)
        plt.xlabel('$\it{x}$', fontproperties=fonten)
        plt.ylabel('$\it{t}$', fontproperties=fonten)
        plt.xticks(fontproperties=fonten)
        plt.yticks(fontproperties=fonten)
        plt.savefig(f'C:/Users/REZ/Desktop/PINN/PINN_{step}.png')
        plt.pause(0.01)
        print('Time:', time.time() - start_time)