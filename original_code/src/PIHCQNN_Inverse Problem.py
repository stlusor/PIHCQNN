import torch
import matplotlib.pyplot as plt
from pennylane import numpy as np
import numpy
import pennylane as qml
import matplotlib as mpl
import time

mpl.rcParams['font.sans-serif']='Times New Roman'
mpl.rcParams['mathtext.fontset']='stix'
fonten = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\Times.ttf', size=14)

### parameter
EA = 1
alpha = 2 * np.pi

def get_derivative(y, x, n):
    if n == 0:
        return y
    else:
        dy_dx = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True, allow_unused=True)[0]
        return get_derivative(dy_dx, x, n - 1)


n_qubits = 5
dev = qml.device("default.qubit.torch", wires=n_qubits)
@qml.qnode(dev)
def qnode(inputs, weights1, weights2, weights3):
    # qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights1, wires=range(n_qubits))
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights2, wires=range(n_qubits))
    # qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    # qml.StronglyEntanglingLayers(weights3, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

# n_layers = 2
# weight_shapes = {"weights": (n_layers, n_qubits)}
weight_shapes = {"weights1": (1, n_qubits, 3), "weights2": (1, n_qubits, 3), "weights3": (1, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

clayer_1 = torch.nn.Linear(1, 5)
clayer_2 = torch.nn.Linear(5, 1)
# clayer_3 =torch.nn.Linear(5, 5)
# softmax = torch.nn.Softmax(dim=1)
layers = [clayer_1, torch.nn.Tanh(), qlayer, torch.nn.Tanh(), clayer_2]
model = torch.nn.Sequential(*layers).cuda()

class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(weight_ini)

        self.reset_params()
    def reset_params(self):
        self.w.data.fill_(2)



    def forward(self, x_f, x_b1, x_b2):
        u = model(x_f)[:, 0].view(-1, 1)
        u_xx = get_derivative(u, x_f, 2)
        res_f = self.w * u_xx + 4 * (torch.pi ** 2) * torch.sin(2 * torch.pi * x_f)
        mse_f_value = torch.mean(torch.pow(res_f, 2))

        mse_b_value = torch.mean(torch.pow(model(x_b1)[:, 0].view(-1, 1), 2)) + torch.mean(torch.pow(model(x_b2)[:, 0].view(-1, 1), 2))

        res_d = torch.sin(2 * torch.pi * x_f).view(-1, 1) - model(x_f)[:, 0].view(-1, 1)
        mse_d_value = torch.mean(torch.pow(res_d, 2))
        print(self.w)
        return mse_f_value + 10 * mse_d_value


x_f = torch.unsqueeze(torch.linspace(0, 1, 50), dim=1)
x_b1 = torch.unsqueeze(torch.linspace(0, 0, 1), dim=1)
x_b2 = torch.unsqueeze(torch.linspace(1, 1, 1), dim=1)
x_f.requires_grad = True
weight_ini = torch.ones(1).requires_grad_(True).cuda()

x_f = x_f.cuda()
x_b1 = x_b1.cuda()
x_b2 = x_b2.cuda()



loss_function = My_loss()
optimizer = torch.optim.Adam([{"params": model.parameters()}, {"params": loss_function.parameters()}], lr=0.005)
print('11111111111111111111111111')
# for name, parameters in (model.parameters() and loss_function.parameters()):
#     print('params name:', name, ':', parameters, parameters.size())
print('11111111111111111111111111')

l2error = []
loss = []
start_time = time.time()
for step in range(20000):
    def closure():
        optimizer.zero_grad()
        loss = loss_function(x_f, x_b1, x_b2)
        # print(loss)
        loss.backward()
        return loss
    optimizer.step(closure)

    if step % 100 == 0:
        loss_tem = loss_function(x_f, x_b1, x_b2)
        print(step, loss_tem)
        loss.append(loss_tem.cpu().data.numpy())
        numpy.savetxt('C:/Users/REZ/Desktop/case1_result/PIHCQNN/loss.txt', loss)
        prediction = model(x_f)[:, 0].view(-1, 1).cpu().data.numpy()
        numpy.savetxt('C:/Users/REZ/Desktop/case1_result/PIHCQNN/prediction.txt', prediction)
        error = prediction - numpy.sin(2 * numpy.pi * x_f.cpu().data.numpy())
        l2_error = numpy.linalg.norm(error) / numpy.linalg.norm(np.sin(2 * np.pi * x_f.cpu().data.numpy()))
        l2error.append(l2_error)
        numpy.savetxt('C:/Users/REZ/Desktop/case1_result/PIHCQNN/l2error.txt', l2error)

        plt.close()
        plt.plot(x_f.cpu().data.numpy(), prediction, c='red', label='Prediction')
        plt.plot(x_f.cpu().data.numpy(), numpy.sin(2 * numpy.pi * x_f.cpu().data.numpy()), c='b', label='Groundtruth')
        plt.title(f'PIHCQNN Prediction', fontproperties=fonten)
        plt.xlabel('$\it{x}$', fontproperties=fonten)
        plt.ylabel('Displacement', fontproperties=fonten)
        plt.xticks(fontproperties=fonten)
        plt.yticks(fontproperties=fonten)
        plt.legend(prop=fonten)
        plt.savefig(f'C:/Users/REZ/Desktop/PIHCQNN.png')
        plt.pause(0.01)
        print('Time:', time.time() - start_time)

        # print(list(model.parameters()))