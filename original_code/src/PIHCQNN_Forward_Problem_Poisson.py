import torch
import matplotlib.pyplot as plt
from pennylane import numpy as np
import numpy
import pennylane as qml
import matplotlib as mpl
import time
import seaborn as sns

torch.cuda.set_device(1)

mpl.rcParams['font.sans-serif']='Times New Roman'
mpl.rcParams['mathtext.fontset']='stix'
fonten = mpl.font_manager.FontProperties(fname='C:\Windows\Fonts\Times.ttf', size=14)

### parameter

def f(x, y):
    u = model(torch.cat((x, y), dim=1))
    u_xx = get_derivative(u, x, 2)
    u_yy = get_derivative(u, y, 2)
    f = u_xx + u_yy - x ** 2 - y ** 2
    return f

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
    # qml.BasicEntanglerLayers(weights1, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights1, wires=range(n_qubits))
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    # qml.BasicEntanglerLayers(weights2, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights2, wires=range(n_qubits))
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.StronglyEntanglingLayers(weights3, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

n_layers = 1
# weight_shapes = {"weights1": (n_layers, n_qubits), "weights2": (n_layers, n_qubits), "weights3": (1, n_qubits, 3)}
weight_shapes = {"weights1": (1, n_qubits, 3), "weights2": (1, n_qubits, 3), "weights3": (1, n_qubits, 3)}
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes).cuda(0)
width = 5
clayer_1 = torch.nn.Linear(2, width)
clayer_2 = torch.nn.Linear(width, n_qubits)
clayer_3 = torch.nn.Linear(n_qubits, width)
clayer_4 = torch.nn.Linear(width, 1)
# softmax = torch.nn.Softmax(dim=1)
layers = [clayer_1, torch.nn.Tanh(), clayer_2, torch.nn.Tanh(), qlayer, torch.nn.Tanh(), clayer_3, torch.nn.Tanh(), clayer_4]
model = torch.nn.Sequential(*layers).cuda()


class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_f, y_f, b_0, b_1, b_2):
        res_f = f(x_f, y_f)
        mse_f_value = torch.mean(torch.pow(res_f, 2))

        pred_b1 = model(torch.cat((b_2, b_0), dim=1))[:, 0].view(-1, 1) - 0.5 * b_2 ** 2
        pred_b2 = model(torch.cat((b_2, b_1), dim=1))[:, 0].view(-1, 1) - 0.5 * b_2 ** 2
        pred_b3 = model(torch.cat((b_0, b_2), dim=1))[:, 0].view(-1, 1) - 0.5 * b_2 ** 2
        pred_b4 = model(torch.cat((b_1, b_2), dim=1))[:, 0].view(-1, 1) - 0.5 * b_2 ** 2
        mse_b_value = torch.mean(torch.pow(pred_b1, 2)) + torch.mean(torch.pow(pred_b2, 2)) + torch.mean(torch.pow(pred_b3, 2)) + torch.mean(torch.pow(pred_b4, 2))

        return mse_f_value + mse_b_value

num = 40
x_f = torch.unsqueeze(torch.linspace(-1, 1, num).repeat(num), dim=1)
y_f_1 = torch.unsqueeze(torch.linspace(-1, 1, num), dim=1)
y_f = torch.unsqueeze(torch.linspace(0, 0, num ** 2), dim=1)
for i in range(num):
    for j in range(num):
        y_f[num * i + j] = y_f_1[i]

b_0 = torch.unsqueeze(torch.linspace(-1, -1, num), dim=1)
b_1 = torch.unsqueeze(torch.linspace(1, 1, num), dim=1)
b_2 = torch.unsqueeze(torch.linspace(-1, 1, num), dim=1)




def grt(x, y):
    return 0.5 * (x * y) ** 2

grtruth = grt(x_f, y_f).data.numpy()


x_f.requires_grad = True
y_f.requires_grad = True
b_0.requires_grad = True
b_1.requires_grad = True
b_2.requires_grad = True
#
x_f = x_f.cuda()
y_f = y_f.cuda()
b_0 = b_0.cuda()
b_1 = b_1.cuda()
b_2 = b_2.cuda()



optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
print(list(model.parameters()))
loss_function = My_loss()
l2error = []
loss = []
start_time = time.time()
for step in range(5000):
    def closure():
        optimizer.zero_grad()
        loss = loss_function(x_f, y_f, b_0, b_1, b_2)
        # print(loss)
        loss.backward()
        return loss
    optimizer.step(closure)

    if step % 50 == 0:
        loss_tem = loss_function(x_f, y_f, b_0, b_1, b_2)
        loss.append(loss_tem.cpu().data.numpy())
        numpy.savetxt('C:/Users/REZ/Desktop/PIHCQNN/loss.txt', loss)
        print(loss_tem)
        prediction = model(torch.cat((x_f, y_f), dim=1))[:, 0].cpu().data.numpy()
        numpy.savetxt('C:/Users/REZ/Desktop/PIHCQNN/prediction.txt', prediction)
        prediction = prediction.reshape(-1, 1)
        error = prediction - grtruth
        l2_error = numpy.linalg.norm(error) / numpy.linalg.norm(grtruth)
        print(step, l2_error)
        l2error.append(l2_error)
        numpy.savetxt('C:/Users/REZ/Desktop/PIHCQNN/l2error.txt', l2error)


        plt.close()
        sns.heatmap(numpy.flipud(np.array(prediction).reshape(num, num).T), cmap='rainbow')
        plt.title(f'PIHCQNN Prediction ({step} Iterations)', fontproperties=fonten)
        plt.xlabel('$\it{x}$', fontproperties=fonten)
        plt.ylabel('$\it{t}$', fontproperties=fonten)
        plt.xticks(fontproperties=fonten)
        plt.yticks(fontproperties=fonten)
        plt.savefig(f'C:/Users/REZ/Desktop/PIHCQNN/PIHCQNN{step}.png')
        plt.pause(0.01)
        print('Time:', time.time() - start_time)

        # torch.save(model, 'C:/Users/REZ/Desktop/PIHCQNN/model.pkl')
        print(model.state_dict())
        torch.save(model.state_dict(), 'C:/Users/REZ/Desktop/PIHCQNN/model_para.pkl')
        # torch.save(model.state_dict(), 'C:/Users/REZ/Desktop/PIHCQNN/model_para.txt')