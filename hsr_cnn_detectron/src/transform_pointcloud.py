import tf2_ros
import tf2_geometry_msgs
import tf
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2



def point_cloud_callback(msg):
    rospy.loginfo("[+] Message Received")
    pc_gen = pc2.read_points(msg)

    try:
        (trans, rot) = listener.lookupTransform("base_link", msg.header.frame_id, rospy.Time(0))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        rospy.logwarn("Failed to get transform")
        return
    transformed_points = []
    for p in pc_gen:
        point = list(p)
        point_transformed = tf.transformations.concatenate_matrices(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot)).dot([point[0], point[1], point[2], 1.0])[:3]
        transformed_points.append(point_transformed)
    transformed_msg = pc2.create_cloud_xyz32(msg.header, transformed_points)
    transformed_msg.header.frame_id='base_link'
    publisher.publish(transformed_msg)


if __name__ == '__main__':
    rospy.init_node('point_cloud_transform')
    listener = tf.TransformListener()
    rospy.Subscriber("/segmented_point_ros", PointCloud2, point_cloud_callback)
    publisher = rospy.Publisher('/transformed_point_cloud', PointCloud2, queue_size=10)
    rospy.spin()