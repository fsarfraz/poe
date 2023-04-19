import rospy
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import h5py
import os
import sys
import copy
import math
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud, CameraInfo, PointCloud2, PointField
import ros_numpy
import glob
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(f'runs1/ScanObjectNN/tensorboard')

test_true = []
test_pred = []

class H5Dataset(Dataset):
    def __init__(self, pnt_cld_array,label_array, num_points):
        self.data = pnt_cld_array[:].astype('float32')
        self.label = label_array[:].astype('int64')
        self.num_points = num_points     
    
    def __getitem__(self, item): 
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        return pointcloud, label
        

    def __len__(self):
        return self.data.shape[0]


def load_data(partition):
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
            continue
        data_mean = np.mean(data, axis=0)
        data -= data_mean
        furthest_distance = np.max(np.sqrt(np.sum(abs(data)**2,axis=-1)))
        data /= furthest_distance
        label = f['label'][:].astype('int64')
        f.close()
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


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNN(nn.Module):
    def __init__(self, output_channels=15):
        super(DGCNN, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(512)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 512, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(512*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=20)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=20)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=20)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=20)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)



def test(data, label=[[2]]):
    
    # pcd = o3d.io.read_point_cloud("test1(1).pcd")
    # pnt_cld = np.asarray(pcd.points) 
    pnt_cld = np.array([data])
    print(pnt_cld.shape, '_____________')
    # pnt_cld = np.asarray(pnt_cld, dtype="float32")
    #pnt_cld = torch.from_numpy(pnt_cld.astype("float32"))
    label = np.array(label)
    # label = np.asarray(label)
    # label = torch.from_numpy(label.astype('long'))
    num_points = 512
    model_path = "/home/r2d2/hsr_rss_project/src/hsr_cnn_detectron/src/checkpoints/dgcnn_2048/models/model.t7"
    test_loader = DataLoader(H5Dataset(pnt_cld_array=pnt_cld,label_array=label,num_points=num_points))
    # test_loader = DataLoader(ModelNet40(num_points=num_points,partition="test"))
    #DataLoader(H5Dataset(pnt_cld_array=pnt_cld,label_array=label,num_points=num_points))


    device = torch.device("cuda")  #"cuda" if args.cuda else 

    #Try to load models
    model = DGCNN().to(device)
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path),strict = False)
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    global test_true
    global test_pred
    # label_check = [[6]]
    # label_check = torch.tensor(label_check)
    for data, labels in test_loader:
        # print("this is labels", labels)
        data, labels = data.to(device), labels.to(device).squeeze()
        print(data.shape)
        # print("this is data",data.dtype)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        print(logits)
        preds = logits.max(dim=1)[1]
        #print(preds)
        test_true.append(labels.cpu().numpy())
        test_pred.append([int(i) for i in preds.detach().cpu().numpy()])
    print("this is tru",test_true)
    print("this is tes",test_pred)
    # test_true = np.concatenate(test_true)
    # test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    with open('pred_detect.csv', 'a+') as f:
        f.writelines(f'{avg_per_class_acc}, {len(test_pred)}\n')
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    print(outstr)


class hsr_dgcnn(object):
    '''
    @To-DO
    '''
    def __init__(self):
        self.rgb_image = None
        self.depth_image = None
        self.pcd = None
        self.rgbd = None
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(0.25)
        self.sub = rospy.Subscriber('/segmented_point_ros', PointCloud2, callback=self.dgcnn)
        self.pointclouds = None


    def dgcnn(self, msg):
        rospy.loginfo('Message Received')
        self.pointclouds = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
        self.pointclouds -= np.mean(self.pointclouds, axis=0)
        # print(self.pointclouds)
        test(self.pointclouds)

    
    def start(self):
        rospy.loginfo('[+] hsr_cnn_detection_node fired!')
        rospy.spin()
        self.bridge = CvBridge()
        while not rospy.is_shutdown():
            self.rate.sleep()


if __name__ == "__main__":
    rospy.init_node('hsr_pointcloud_dgcnn', anonymous=True)
    dgcnn_node = hsr_dgcnn()
    dgcnn_node.start()
