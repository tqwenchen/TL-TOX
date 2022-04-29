import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as opt
import models
from models import dvib
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from utils import normalization, make_tensor, progress_bar
import scipy.io as sio
import csv
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='ToxIBTL Training')
parser.add_argument('--model', default="rcnn1", type=str,
                    help='model type (default: rcnn1)')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=100, type=int,
                    help='total epochs to run')
parser.add_argument('--epoch', default=500, type=int,
                    help='total epochs to run')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# loss_dir
loss_log_dir = os.path.join('tensorboard', 'loss')
loss_writer = SummaryWriter(log_dir=loss_log_dir)

# train_log_dir
train_log_dir = os.path.join('tensorboard', 'train')
train_writer = SummaryWriter(log_dir=train_log_dir)

# test_log_dir
test_log_dir = os.path.join('tensorboard', 'test')
test_writer = SummaryWriter(log_dir=test_log_dir)


#set random seed
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


train_path = 'data/protein_train1002.csv'
test_path = 'data/protein_test1002.csv'

train_data = sio.loadmat('data/benchmark/protein_train.mat')
train_FEGS = torch.Tensor(normalization(train_data['FV']))
print("shape of train FEGS: ", train_FEGS.shape) #torch.Size([10084, 578])

test_data = sio.loadmat('data/benchmark/protein_test.mat')
test_FEGS = torch.Tensor(normalization(test_data['FV']))
print("shape of test FEGS: ", test_FEGS.shape) #torch.Size([729, 578]

train_pssm, train_len, train_label = make_tensor(train_path)
# print("shape of train_pssm", train_pssm.shape) #torch.Size([10084, 1002, 20])
test_pssm, test_len,test_label = make_tensor(test_path)

train_data = DataLoader(TensorDataset(train_pssm, train_len,train_FEGS,train_label), batch_size=100, shuffle=True)
test_data = DataLoader(TensorDataset(test_pssm, test_len,test_FEGS, test_label), batch_size=100)

print("data done")


CE = nn.CrossEntropyLoss(reduction='sum')
betas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6,1e-7]


def calc_loss(y_pred, labels, enc_mean, enc_std, beta=1e-3):
    """
    y_pred : [batch_size,2]
    label : [batch_size,1]
    enc_mean : [batch_size,z_dim]
    enc_std: [batch_size,z_dim]
    """

    ce = CE(y_pred, labels)
    KL = 0.5 * torch.sum(enc_mean.pow(2) + enc_std.pow(2) - 2 * enc_std.log() - 1)

    return (ce + beta * KL) / y_pred.shape[0]


beta = 2#0，1，3，4，5，6


start_epoch = 0
best_acc = 0
#Model
print('==> Building model..')
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth' + args.name + '_'
                            + str(args.seed))
    # net.load_state_dict(checkpoint['net'])
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    rng_state = checkpoint['rng_state']
    torch.set_rng_state(rng_state)
else:
    print('==> Building model..')
    net = models.__dict__[args.model]()
net = net.to(device)
if not os.path.isdir('results'):
    os.mkdir('results')
logname = ('results/log' +  '_' + args.model + '_epoch500_rcnn_'
           + str(args.seed) + '.csv')


optimizer = opt.Adam(net.parameters(),lr = args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_correct = 0
    if epoch % 10 == 0 and epoch > 0:
        scheduler.step()

    for batch_idx, (sequences, lengths, FEGS, labels) in enumerate(train_data):
        seq_lengths, perm_idx = lengths.sort(dim=0, descending=True)
        seq_tensor = sequences[perm_idx].to(device)
        FEGS_tensor = FEGS[perm_idx].to(device)
        label = labels[perm_idx].long().to(device)
        y_pred, end_means, enc_stds, latent = net(seq_tensor, seq_lengths, FEGS_tensor)
        loss = calc_loss(y_pred, label, end_means, enc_stds, betas[beta])
        _, train_pred = torch.max(y_pred, 1)
        train_correct += train_pred.eq(label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    loss_writer.add_scalar("train loss", train_loss, epoch)

    train_writer.add_scalar("train acc", 100. * train_correct / len(train_data.dataset), epoch)

    progress_bar(batch_idx, len(train_data),
                 'Loss: %.6f | Acc: %.4f%% (%d/%d)'
                 % (train_loss,
                    100. * train_correct / len(train_data.dataset), train_correct, len(train_data.dataset)))

    print("Train Epoch:{} Average loss: {:.6f} ACC:{}/{} ({:.4f}%)".format(
        epoch,
        train_loss,
        train_correct, len(train_data.dataset),
        100. * train_correct / len(train_data.dataset)))

    return (train_loss, 100 * train_correct / len(train_data.dataset))

def test(epoch):
    global best_acc
    correct = 0
    # total = 0
    y_pre = []
    y_test = []
    with torch.no_grad():
        for batch_idx, (sequences, lengths, FEGS, labels) in enumerate(test_data):
            seq_lengths, perm_idx = lengths.sort(dim=0, descending=True)
            seq_tensor = sequences[perm_idx].to(device)
            FEGS_tensor = FEGS[perm_idx].to(device)
            label = labels[perm_idx].long().to(device)
            y_test.extend(label.cpu().detach().numpy())
            y_pred, end_means, enc_stds, latent = net(seq_tensor, seq_lengths, FEGS_tensor)
            y_pre.extend(y_pred.argmax(dim=1).cpu().detach().numpy())

            _, pred = torch.max(y_pred, 1)
            # total += len(test_data.dataset)
            correct += pred.eq(label).sum().item()
        test_writer.add_scalar("test acc", 100. * correct / len(test_data.dataset), epoch)
        progress_bar(batch_idx, len(test_data),
                     'Test Acc: %.4f%% (%d/%d)'
                     %(100. * correct / len(test_data.dataset),
                         correct, len(test_data.dataset)))

        print('\nTest: Accuracy:{}/{} ({:.4f}%) f1:({:.4f}%) mcc:({:.4f}%)\n'.format(
            correct, len(test_data.dataset),
            100. * correct / len(test_data.dataset),
            metrics.f1_score(y_test, y_pre),
            metrics.matthews_corrcoef(y_test, y_pre)
        ))
    # Save Checkpoint
    acc = 100 * correct / len(test_data.dataset)
    if acc > best_acc:
        print('Saving..')
        state = {
            # 'net': net.state_dict(),
            'net': net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint/RCNN'):
            os.mkdir('checkpoint/RCNN')

        torch.save(state, './checkpoint/RCNN/ckpt.pth_' + args.model + '_epoch500_rcnn' + '_'
                   + str(args.seed))
        best_acc = acc
    return (100. * correct / len(test_data.dataset),metrics.f1_score(y_test, y_pre),metrics.matthews_corrcoef(y_test, y_pre))

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train_loss', 'train acc', 'test acc', 'f1_score', 'matthews_corrcoef'])

for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc = train(epoch)
    # train_writer.add_scalar("train acc", train_acc, epoch)
    test_acc,f1_score,matthews_corrcoef = test(epoch)
    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss, train_acc, test_acc, f1_score, matthews_corrcoef])

loss_writer.close()
train_writer.close()
test_writer.close()