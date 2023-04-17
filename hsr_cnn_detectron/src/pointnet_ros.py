import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import h5py
import sklearn.metrics as metrics
import open3d as o3d

# train_h5 = sorted(glob.glob('./training*.h5'))
# test_h5 = sorted(glob.glob('./test*.h5'))


pnt_cld = None
label = None


class H5Dataset(Dataset):
    def __init__(self, pnt_cld_array,label_array, num_points):
        # pnt_cld_array = np.array([[pnt_cld_array[0]], [pnt_cld_array[0]]])
        # label = np.array([[label[0]], [[0]]])
        self.data = np.array([pnt_cld_array[:].astype('float32')])
        # self.data_mean = np.mean(self.data, axis=0)
        # self.data -= self.data_mean
        self.furthest_distance = np.max(np.sqrt(np.sum(abs(self.data)**2,axis=-1)))
        self.data /= self.furthest_distance
        self.label = label_array[:].astype('int64')
        self.num_points = num_points     
        # self.data = np.concatenate(self.data, axis=0)
        # self.label = np.concatenate(self.label, axis=0)
        print(self.data.shape, '+++++++________+++++++')
    
    def __getitem__(self, item): 
        pointcloud = self.data[item][:self.num_points]
        # print("this is pointcloud", pointcloud)
        label = self.label[item]
        return pointcloud, label
        

    def __len__(self):
        return self.data.shape[0]    
    


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        # os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'scanobjectnn', '%s_objectdataset*.h5'%partition)):
        f = h5py.File(h5_name, 'r')
        # print(f)
        try:
            data = f['data'][:].astype('float32')
        except Exception as e:
            print(h5_name, '********')
            continue
        data_mean = np.mean(data, axis=0)
        data -= data_mean
        furthest_distance = np.max(np.sqrt(np.sum(abs(data)**2,axis=-1)))
        data /= furthest_distance
        label = f['label'][:].astype('int64')
        f.close()
        print(data.shape)
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='training'):
        self.data, self.label = load_data(partition)
        print(self.label)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    # for data, label in train:
    #     print(data.shape)
    #     print(label.shape)


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
        


class PointNet(nn.Module):
    def __init__(self, output_channels=15):
        super(PointNet, self).__init__()
        # self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, 1024, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze().view(1, 1024)
        # x = F.adaptive_max_pool1d(x, 1).squeeze()
        # print(x.shape, '+++++++++++ZZZZZZZZZ')
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x
    
def train():
    train_loader = DataLoader(ModelNet40(partition='training', num_points=1024), num_workers=4,
                              batch_size=130, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=1024), num_workers=4,
                             batch_size=130, shuffle=True, drop_last=False)

    device = torch.device("cuda")

    #Try to load models
    # if args.model == 'pointnet':
    model = PointNet().to(device)
    # else:
    #     raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if True:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=0.001*100, momentum=0.0, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, 20, eta_min=0.001)
    
    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(20):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 metrics.accuracy_score(
                                                                                     train_true, train_pred),
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        # io.cprint(outstr)
        print(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        # io.cprint(outstr)
        print(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'model.t7')


def test():
    # test_loader = DataLoader(ModelNet40(partition='test', num_points=1024),
    #                          batch_size=1, shuffle=False, drop_last=False)
    global label
    global pnt_cld
    test_loader = DataLoader(H5Dataset(pnt_cld_array=pnt_cld,label_array=label,num_points=1024))
    device = torch.device("cuda")
    #Try to load models
    model = PointNet().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('/home/r2d2/hsr_rss_project/src/hsr_cnn_detectron/src/model.t7'))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        # print(data.shape, '++++++++++++++')
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
        # print(test_true, test_pred, end='\n\n')
    # test_true = np.concatenate(test_true)
    # test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    # io.cprint(outstr)
    print(outstr)
    

if __name__ == '__main__':
    mesh = o3d.io.read_triangle_mesh('/home/r2d2/hsr_rss_project/src/hsr_cnn_detectron/src/006_mustard_bottle_textured.obj')
    pnt_cld = mesh.sample_points_poisson_disk(1024)
    pnt_cld = np.asarray(pnt_cld.points)
    label = np.array([[6]])
    # train()
    # train()