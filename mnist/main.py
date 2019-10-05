from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from myia import myia, value_and_grad
from myia.ir import sexp_to_node
from myia.lib import setter_from_getter
from myia.abstract import Macro, build_value, macro
from myia.frontends import activate_frontend

from myia.debug import traceback

activate_frontend('pytorch')

from myia.operations import primitives as P
from dataclasses import dataclass
import numpy as np


@dataclass
class Optimizer:
    parameter_names: tuple
    lr: float

    def __init__(self, parameter_names, lr):
        self.parameter_names = tuple(tuple(_p[0].split('.'))
                                              for _p in parameter_names)
        self.lr = np.float32(lr)
        self.momentum = 0.5

    def update_rule(self, p, g):
        return p - self.lr * g

    def __call__(self, model, dmodel):
        return self.update(self.parameter_names, model, dmodel, self.update_rule)

    @macro
    async def update(info, self_ref, param_names_ref, model_ref, dmodel_ref, update_rule_ref):
        param_names_cst = build_value(await param_names_ref.get())
        #self = build_value(await self_ref.get())
        new_model = model_ref.node
        dmodel = dmodel_ref.node
        update_rule = update_rule_ref.node
        model_abs = await model_ref.get()

        for k in param_names_cst:
            p = new_model
            g = dmodel
            for c in k:
                p = (P.record_getitem, p, c)
                g = (P.record_getitem, g, c)

            p_node = sexp_to_node(p, info.graph)
            g_node = sexp_to_node(g, info.graph)

            pn = info.graph.apply(update_rule, p_node, g_node)

            new_model = sexp_to_node(setter_from_getter(p, pn), info.graph)

        return new_model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        #x = x.view(-1, 4*4*50)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

backend = 'pytorch'
backend_options = {'device': 'cpu'}

def cost(model, data, target):
    output = model(data)
    return F.nll_loss(output, target)


@myia(backend=backend, backend_options=backend_options, specialize_values=["model", "optimizer"])
def step(model, data, target, optimizer):
    loss, dmodel = value_and_grad(cost, 'model')(model, data, target)
    return loss, optimizer(model, dmodel)
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        """
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #"""
        loss, model = step(model, data, target,  optimizer)
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return model, optimizer


@myia(backend=backend, backend_options=backend_options, specialize_values=["model"])
def step_eval(model, data, target):
    output = model(data)
    test_loss = F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
    correct = pred.eq(target.view_as(pred)).sum().item()
    return test_loss, correct

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            """
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            test_loss = test_loss + F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            correct = correct + pred.eq(target.view_as(pred)).sum().item()
            """
            test_loss_iter, correct_iter = step_eval(model, data, target)
            test_loss = test_loss + test_loss_iter
            correct = correct + correct_iter

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = Optimizer(model.named_parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model, optimizer = train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")
        
if __name__ == '__main__':
    main()
