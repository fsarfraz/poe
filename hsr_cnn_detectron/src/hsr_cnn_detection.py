'''
@Implementation of MASK-RCNN using Detectron2 for RSS Final Project
@authors - Sarvesh and Farhan
@affiliation - Northeastern University

To-Do --> Rewrite implementation using U-Net rather than detectron2.
^^^~~~ I don't know if I'll able to achieve this in the given time frame...

This script is a helper script that semantically segments object in 2D, 
extracts the point cloud from that region and publishes the pointnet, 
which are further subscribed by object detection methods like pointnet,
pointnet++ and DGCNN.

What works for now--
Segmenting the desired object - works
Generating point cloud of desired object - works
Able to publish point cloud - works but need to work with transforms!!!
_____________________________________________________________________
_____________________________________________________________________
_____________________________________________________________________
_____________________________________________________________________
'''
#!/usr/bin/python3
# ^~ Shebang line, change it as per your system
# can be found using which command in terminal `which python3`

# v~ Necessary imports for the project
import numpy as np
import cv2
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud, CameraInfo, PointCloud2, PointField
import rospy
import torch, detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, random
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import open3d as o3d
from open3d_ros_helper import open3d_ros_helper as orh
import ros_numpy
import sys


class hsr_cnn_detection(object):
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
        self.segment_publisher = rospy.Publisher('segmented', Image, queue_size=10)
        self.point_publisher = rospy.Publisher('segmented_point_ros', PointCloud2, queue_size=10)
        self.image_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=100, slop=0.02)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=100, slop=0.02)
        self.ts.registerCallback(self.project_2d_3d)
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.fx = 533.8970730178461
        self.fy = 534.3109677231259
        self.cx = 321.0284419169324
        self.cy = 241.1102341748379
        self.pinhole_camera_intrinsic.set_intrinsics(640, 480, 533.8970730178461, 534.3109677231259, 321.0284419169324, 241.1102341748379)
        self.detectron_cfg = get_cfg()
        self.detectron_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.detectron_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.detectron_predictor = None
        self.detectron_output = None
        self.detectron_visualizer = None
        x = PointField()
        self.boxes = None
        self.mask = None
        self.depth_image_x = None
        self.x_cam = 0
        self.y_cam = 0
        self.z_cam = 0
        self.all_point_calc = None
        # self.FIELDS_XYZ = [
        #     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        #     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        # ]
        # self.FIELDS_XYZRGB = self.FIELDS_XYZ + \
        #     [PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)]


    def project_2d_3d(self, image_msg, depth_msg):
        rospy.loginfo('Message Received')
        # self.rgb_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        self.rgb_image = self.bridge.imgmsg_to_cv2(image_msg)
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        self.depth_image_x = self.bridge.imgmsg_to_cv2(depth_msg)
        # self.segment_publisher.publish(self.bridge.cv2_to_imgmsg(self.depth_image))
        try:
            self.detectron_predictor = DefaultPredictor(self.detectron_cfg)
            self.detectron_output = self.detectron_predictor(self.rgb_image)
            self.boxes = self.detectron_output['instances'][self.detectron_output['instances'].pred_classes == 39].pred_boxes
            self.boxes = list(self.boxes)[0].detach().cpu().numpy()
            self.mask = self.detectron_output['instances'][self.detectron_output['instances'].pred_classes == 39].pred_masks
            self.mask = list(self.mask)[0].detach().cpu().numpy()
            self.mask = cv2.cvtColor((self.mask*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.rgb_image = cv2.bitwise_and(self.rgb_image, self.mask)
            self.mask = self.mask.astype(np.uint16)
            self.depth_image = cv2.bitwise_and(self.depth_image, self.mask[:,:,0])
            ### Something that is not required for now, and computation time can be decreased!
            # self.all_point_calc = []
            # for idx,pixel in enumerate(self.depth_image):
            #     for id, pix in enumerate(pixel):
            #         if pix != 0:
            #             self.all_point_calc.append([self.fx*(id/pix)+self.cx, self.fy*(id/pix)+self.cy , pix])
            # self.all_point_calc = np.asarray(self.all_point_calc)
            # apcd = o3d.geometry.PointCloud()
            # apcd.points = o3d.utility.Vector3dVector(self.all_point_calc)
            # self.point_publisher.publish(self.o3d_to_pointcloud2(apcd, 'segmented_point_ros'))
            (x, y) = (int(self.boxes[0]), int(self.boxes[1]))
            (w, h) = (int(self.boxes[2])-int(self.boxes[0]), int(self.boxes[3])-int(self.boxes[1]))
            y_mid = (int(self.boxes[0]) + int(self.boxes[2])) // 2
            x_mid = (int(self.boxes[1]) + int(self.boxes[3])) // 2
        except Exception as e:
            print('No bottle', e)
            (x, y) = (0,0)
            (w, h) = (0,0)
            x_mid = 0
            y_mid = 0
        # self.mask = np.reshape(self.mask*255, (640, 480)).astype(np.uint8)
        
        # cv2.imshow('image', self.mask)
        # for x in self.mask:
        #     print(x)
        # print(self.mask.shape)
        # print(self.boxes)
        
        # self.rgb_image = self.rgb_image[100:500, 100:500]
        # self.depth_image = self.depth_image[100:500, 100:500]
        # self.pinhole_camera_intrinsic.set_intrinsics(400, 400, 533.8970730178461, 534.3109677231259, 321.0284419169324, 241.1102341748379)
        # self.rgb_image = cv2.resize(self.rgb_image, (640, 480), interpolation= cv2.INTER_LINEAR)
        # self.depth_image = cv2.resize(self.depth_image, (640, 480), interpolation= cv2.INTER_LINEAR)
        # self.rgb_image = self.rgb_image
        # self.depth_image = self.depth_image
        # self.rgb_image[x_mid-5:x_mid+5, y_mid-5:y_mid+5] = 255
        
        self.segment_publisher.publish(self.bridge.cv2_to_imgmsg(self.rgb_image))
        self.rgb_image = o3d.geometry.Image(self.rgb_image)
        self.depth_image = o3d.geometry.Image(self.depth_image)
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(self.rgb_image, self.depth_image, convert_rgb_to_intensity=False)
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd, self.pinhole_camera_intrinsic)
        # self.mask = np.array(self.mask * 255).astype('uint8')
        # print(self.pcd)
        # print(np.asarray(self.depth_image_x, dtype=np.float64)[x_mid, y_mid])
        '''
        Rectify this formula to and test with o3d -
        ((u - c_x) * d) / f_x
        ((v - c_y) * d) / f_y
        '''
        self.z_cam = np.asarray(self.depth_image_x, dtype=np.float64)[x_mid, y_mid]
        self.x_cam = self.fx*(x_mid / self.z_cam)+self.cx
        self.y_cam = self.fy*(y_mid / self.z_cam)+self.cy
        print(self.x_cam, self.y_cam, self.z_cam)
        self.pcd = self.pcd.transform(([1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]))
        pcd_numpy = np.asarray(self.pcd.points)
        self.pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
        o3d.io.write_point_cloud('test1.pcd', self.pcd)
        self.point_publisher.publish(self.o3d_to_pointcloud2(self.pcd, 'segmented_point_ros'))
        # sys.exit(0)
        # print(self.pcd.points)
        # o3d.visualization.draw_geometries([self.pcd])
        

    def o3d_to_pointcloud2(self, pcd, frame_id='frasier'):
        # pcd_numpy = np.asarray(pcd)
        # ros_dtype = PointField.FLOAT32
        # dtype = np.float32
        # itemsize = np.dtype(dtype).itemsize
        # data = pcd_numpy.astype(dtype=dtype).tobytes()
        # fields = [PointField(name=n, offset=i*itemsize, datatype=ros_dtype, count=1) for i,n in enumerate('xyz')]
        pc_o3d = orh.o3dpc_to_rospc(pcd, frame_id=frame_id, stamp=rospy.Time.now())
        return pc_o3d
    
    def start(self):
        rospy.loginfo('[+] hsr_cnn_detection_node fired!')
        rospy.spin()
        self.bridge = CvBridge()
        while not rospy.is_shutdown():
            self.rate.sleep()



if __name__ == '__main__':
    rospy.init_node('hsr_cnn_detection_node', anonymous=True)
    cnn_node = hsr_cnn_detection()
    cnn_node.start()
    