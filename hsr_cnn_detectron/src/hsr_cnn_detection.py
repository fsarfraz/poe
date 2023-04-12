'''
@Implementation of MASK-RCNN using Detectron2 for RSS Final Project
@authors - Sarvesh and Farhan
@affiliation - Northeastern University

To-Do --> Rewrite implementation using U-Net rather than detectron2.

This script is a helper script that semantically segments object in 2D, 
extracts the point cloud from that region and publishes the pointnet, 
which are further subscribed by object detection methods like pointnet,
pointnet++ and DGCNN.
'''
#!/usr/bin/python3
# ^~ Shebang line, change it as per your system
# can be found using which command in terminal `which python3`

# v~ Necessary imports for the project
import numpy as np
import cv2
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud, CameraInfo
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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
        self.loop_rate = rospy.Rate(1)
        self.segment_publisher = rospy.Publisher('segmented', Image, queue_size=10)
        self.point_publisher = rospy.Publisher('segmented_point', PointCloud, queue_size=10)
        self.image_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/hsrb/head_rgbd_sensor/depth_registered/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=100, slop=0.02)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=100, slop=0.02)
        self.ts.registerCallback(self.project_2d_3d)
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.pinhole_camera_intrinsic.set_intrinsics(640, 480, 533.8970730178461, 534.3109677231259, 321.0284419169324, 241.1102341748379)
        self.detectron_cfg = get_cfg()
        self.detectron_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.detectron_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.detectron_predictor = None
        self.detectron_output = None
        self.detectron_visualizer = None


    def project_2d_3d(self, image_msg, depth_msg):
        rospy.loginfo('Message Received')
        self.rgb_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='8UC1')
        # self.segment_publisher.publish(self.bridge.cv2_to_imgmsg(self.depth_image))
        self.detectron_predictor = DefaultPredictor(self.detectron_cfg)
        self.detectron_output = self.detectron_predictor(self.rgb_image)
        # rospy.loginfo(self.detectron_output['instances'])
        self.detectron_visualizer = Visualizer(self.rgb_image[:, :, ::-1], MetadataCatalog.get(self.detectron_cfg.DATASETS.TRAIN[0]), scale=1.2)
        self.detectron_output = self.detectron_visualizer.draw_instance_predictions(self.detectron_output['instances'][self.detectron_output['instances'].pred_classes == 39].to('cpu'))
        # [self.detectron_output['instances'].pred_classes == 31]
        self.segment_publisher.publish(self.bridge.cv2_to_imgmsg(self.detectron_output.get_image()[:, :, ::-1]))
        # plt.imshow(self.rgb_image)
        # plt.show()
        # cv2.imshow('x', self.rgb_image)
        self.rgb_image = self.rgb_image/255.0
        self.rgb_image = o3d.geometry.Image(self.rgb_image.astype(np.float32))
        self.depth_image = o3d.geometry.Image(self.depth_image)
        self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(self.rgb_image, self.depth_image)
        self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd, self.pinhole_camera_intrinsic)
        self.point_publisher.publish()
        # o3d.visualization.draw_geometries([self.pcd])
        

    
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
    