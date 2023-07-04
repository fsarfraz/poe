'''
@Implementation of MASK-RCNN using Detectron2 for RSS Final Project
@authors - Sarvesh Prajapati & Farhan Sarfraz
@email - sarvesh101p@gmail.com, far.sarfraz98@gmail.com
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
import tf2_ros
import tf2_geometry_msgs
import tf

class hsr_cnn_detection(object):
    '''
    @To-DO
    '''
    def __init__(self, rgb_topic, depth_topic, base_link, fx, fy, cx, cy):
        self.rgb_image = None
        self.depth_image = None
        self.pcd = None
        self.rgbd = None
        self.base_link = base_link
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.bridge = CvBridge()
        self.loop_rate = rospy.Rate(0.25)
        self.segment_publisher = rospy.Publisher('segmented', Image, queue_size=10)
        self.point_publisher = rospy.Publisher('segmented_point_ros', PointCloud2, queue_size=10)
        self.image_sub = message_filters.Subscriber(rgb_topic, Image)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub], queue_size=100, slop=0.02)
        self.ts.registerCallback(self.project_2d_3d)
        self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        self.pinhole_camera_intrinsic.set_intrinsics(640, 480, fx, fy, cx, cy)
        self.detectron_cfg = get_cfg()
        self.detectron_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.detectron_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.detectron_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.detectron_predictor = None
        self.detectron_output = None
        self.detectron_visualizer = None
        self.boxes = None
        self.mask = None
        self.depth_image_x = None
        self.x_cam = 0
        self.y_cam = 0
        self.z_cam = 0
        self.all_point_calc = None
        self.mask_ = None



    def project_2d_3d(self, image_msg, depth_msg):
        rospy.loginfo('Message Received')
        self.rgb_image = self.bridge.imgmsg_to_cv2(image_msg)
        self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='16UC1')
        try:
            self.detectron_predictor = DefaultPredictor(self.detectron_cfg)
            self.detectron_output = self.detectron_predictor(self.rgb_image)
            print(self.detectron_output['instances'].pred_classes)
            self.boxes = self.detectron_output['instances'][self.detectron_output['instances'].pred_classes == 39].pred_boxes
            self.boxes = list(self.boxes)[0].detach().cpu().numpy()
            self.mask = self.detectron_output['instances'][self.detectron_output['instances'].pred_classes == 39].pred_masks
            self.mask = list(self.mask)[0].detach().cpu().numpy()
            self.mask_ = self.mask
            self.mask = cv2.cvtColor((self.mask).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR)
            self.rgb_image = cv2.bitwise_and(self.rgb_image, self.mask)
            self.mask_ = cv2.cvtColor((self.mask_).astype(np.uint16)*65535, cv2.COLOR_GRAY2BGR)
            self.depth_image = cv2.bitwise_and(self.depth_image, self.mask_[:,:,0])
           
        except Exception as e:
            rospy.logwarn(e)
            return
        try:
            self.segment_publisher.publish(self.bridge.cv2_to_imgmsg(self.rgb_image))
            self.rgb_image = o3d.geometry.Image(self.rgb_image)
            self.depth_image = o3d.geometry.Image(self.depth_image)
            self.rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(self.rgb_image, self.depth_image, convert_rgb_to_intensity=False)
            self.pcd = o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd, self.pinhole_camera_intrinsic)
            
            pcd_numpy = np.asarray(self.pcd.points)
            self.pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
            # o3d.io.write_point_cloud('test1.pcd', self.pcd)
            self.point_publisher.publish(self.o3d_to_pointcloud2(self.pcd, self.base_link))
        except Exception as e:
            rospy.logwarn(e)
        # sys.exit(0)
        # print(self.pcd.points)
        # o3d.visualization.draw_geometries([self.pcd])
        

    def o3d_to_pointcloud2(self, pcd, frame_id='head_rgbd_sensor_rgb_frame'):
        pc_o3d = orh.o3dpc_to_rospc(pcd, frame_id=frame_id, stamp=rospy.Time.now())
        return pc_o3d
    
    def start(self):
        rospy.loginfo('[+] hsr_cnn_detection_node fired!')
        rospy.spin()
        self.bridge = CvBridge()
        while not rospy.is_shutdown():
            self.rate.sleep()



if __name__ == '__main__':
    rgb_topic = None
    depth_topic = None
    base_link = None
    fx = None
    fy = None
    cx = None
    cy = None
    if len(sys.argv) < 3:
        rospy.logerr("Enough Arguments not provided, check launch file or contact the author.")
    else:
        rgb_topic = sys.argv[1]
        depth_topic = sys.argv[2]
        base_link = sys.argv[3]
        fx = np.float64(sys.argv[4])
        fy = np.float64(sys.argv[5])
        cx = np.float64(sys.argv[6])
        cy = np.float64(sys.argv[7])
    rospy.init_node('hsr_cnn_detection_node', anonymous=True)
    cnn_node = hsr_cnn_detection(rgb_topic=rgb_topic, depth_topic=depth_topic, base_link=base_link, fx=fx, fy=fy, cx=cx, cy=cy)
    cnn_node.start()
