/*
  velo2cam_calibration - Automatic calibration algorithm for extrinsic parameters of a stereo camera and a velodyne
  Copyright (C) 2017-2018 Jorge Beltran, Carlos Guindel

  This file is part of velo2cam_calibration.

  velo2cam_calibration is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.

  velo2cam_calibration is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with velo2cam_calibration.  If not, see <http://www.gnu.org/licenses/>.
*/

/*
  laser_pattern: Find the circle centers in the laser cloud
*/

#define PCL_NO_PRECOMPILE

#include <ros/ros.h>
#include "ros/package.h"
#include <sensor_msgs/PointCloud2.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_msgs/PointIndices.h>
#include <pcl_msgs/ModelCoefficients.h>
#include <pcl/common/geometry.h>
#include <pcl/common/eigen.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/impl/passthrough.hpp>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <dynamic_reconfigure/server.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <velo2cam_calibration/LaserConfig.h>
#include "velo2cam_utils.h"
#include <velo2cam_calibration/ClusterCentroids.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace sensor_msgs;

ros::Publisher cumulative_pub, centers_pub, pattern_pub2, range_pub,
               coeff_pub, aux_pub, auxpoint_pub, debug_pub, edges_cloud_pub, inliers_pub, pattern_pub;
int nFrames; // Used for resetting center computation
int n_rings;
pcl::PointCloud<pcl::PointXYZ>::Ptr cumulative_cloud;

// Dynamic parameters
double plane_distance_threshold_,edge_range_threshold;
double passthrough_radius_min_, passthrough_radius_max_, circle_radius_, circle_radius_threshold_,
       centroid_distance_min_, centroid_distance_max_, min_distance_between_centers_;
Eigen::Vector3f axis_;
double angle_threshold_;
double cluster_size_;
int clouds_proc_ = 0, clouds_used_ = 0;
int min_centers_found_;

void readCloudToMsg(const sensor_msgs::PointCloud2 &cloud, pcl::PointCloud<Velodyne::Point> &pcl_cloud)
{
  sensor_msgs::PointCloud2ConstIterator<float> it_x(cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> it_y(cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> it_z(cloud, "z");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> it_r(cloud, "ring");

  Velodyne::Point laser_pt;

  for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z, ++it_r) {
    laser_pt.x = *it_x;
    laser_pt.y = *it_y;
    laser_pt.z = *it_z;
    laser_pt.ring = *it_r;
    // skip NaN and INF valued points
    if (pcl_isfinite(laser_pt.x) && pcl_isfinite(laser_pt.y) && pcl_isfinite(laser_pt.z)) {
      pcl_cloud.push_back(laser_pt);
    }
    else
      continue;
  }
}

inline float squaredDist(Velodyne::Point* a, Velodyne::Point*b){ return pow(a->x - b->x, 2) + pow(a->y - b->y, 2) + pow(a->z - b->z, 2);}

void calculateDistanceFromEachOther(const vector<Velodyne::Point*>& v, int& i1, int& i2)
{
  float maxdist = 0;
  float d = 0;
  for (int i=0; i<v.size() - 1;i++)
  {
    for(int j=i+1;j<v.size();j++)
    {
      d = squaredDist(v[i],v[j]);
      if (d > maxdist)
      {
        maxdist = d;
        i1 = i;
        i2 = j;
      }
    }
  }
}


bool detectCirclesInPointcloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, std::vector<pcl::PointXYZ>& centers)
{
  if (cloud->empty())
    return false;

  /// cloud to image
  cv::Mat inliner_plane;
  pcl::PointXYZ min_pt,max_pt;
  pcl::getMinMax3D(*cloud,min_pt,max_pt);
  // set image size
  double dx = max_pt.x - min_pt.x;
  double dy = max_pt.y - min_pt.y;
  double scale = 100;
  inliner_plane = cv::Mat(cvRound(scale*dy)+1,cvRound(scale*dx)+1,CV_8UC1);

  /// fill image
  for (auto& pt : *cloud) {
    // column, row
    cv::Point image_point(cvRound(scale * (pt.x - min_pt.x)), cvRound(scale * (pt.y - min_pt.y)));
    inliner_plane.at<uchar>(image_point) = 255;
  }

  /// blur to fill gaps
  cv::Mat closed;
  cv::morphologyEx(inliner_plane, closed, cv::MORPH_DILATE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)), cv::Point(-1,-1), 3);
  cv::GaussianBlur(closed,closed,cv::Size(9,9),2,2);

  /// detect circle centers
  std::vector<cv::Vec3f> circles;
  cv::HoughCircles(closed, circles, cv::HOUGH_GRADIENT, 1, cvRound(min_distance_between_centers_*scale), 150, 10, cvRound((circle_radius_-circle_radius_threshold_)*scale), cvRound((circle_radius_+circle_radius_threshold_)*scale));
  for( size_t i = 0; i < circles.size(); i++ )
  {
    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    double x = center.x / scale + min_pt.x;
    double y = center.y / scale + min_pt.y;
    // all z values are identical anyways
    centers.push_back(pcl::PointXYZ(x, y, max_pt.z));
  }
  return centers.size() >= min_centers_found_ && centers.size() < 5;
}

void callback(const PointCloud2::ConstPtr& laser_cloud){

  if(DEBUG) ROS_INFO("[Laser] Processing cloud...");

  pcl::PointCloud<Velodyne::Point>::Ptr velocloud (new pcl::PointCloud<Velodyne::Point>),
                                        velo_filtered(new pcl::PointCloud<Velodyne::Point>),
                                        pattern_cloud(new pcl::PointCloud<Velodyne::Point>);

  if (laser_cloud->width == 0 || laser_cloud->height == 0)
    return;

  clouds_proc_++;
  readCloudToMsg(*laser_cloud, *velocloud);

  Velodyne::addRange(*velocloud); // For latter computation of edge detection

  pcl::PointCloud<Velodyne::Point>::Ptr radius(new pcl::PointCloud<Velodyne::Point>);
  pcl::PassThrough<Velodyne::Point> pass2;
  pass2.setInputCloud (velocloud);
  pass2.setFilterFieldName ("range");
  pass2.setFilterLimits (passthrough_radius_min_, passthrough_radius_max_);
  pass2.filter (*velo_filtered);

  sensor_msgs::PointCloud2 range_ros;
  pcl::toROSMsg(*velo_filtered, range_ros);
  range_ros.header = laser_cloud->header;
  range_pub.publish(range_ros);

  // Plane segmentation
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

  pcl::SACSegmentation<Velodyne::Point> plane_segmentation;
  plane_segmentation.setModelType (pcl::SACMODEL_PARALLEL_PLANE);
  plane_segmentation.setDistanceThreshold (plane_distance_threshold_);
  plane_segmentation.setMethodType (pcl::SAC_PROSAC);
  plane_segmentation.setAxis(Eigen::Vector3f(axis_[0], axis_[1], axis_[2]));
  plane_segmentation.setEpsAngle (angle_threshold_);
  plane_segmentation.setOptimizeCoefficients (true);
  plane_segmentation.setMaxIterations(1000);
  plane_segmentation.setInputCloud (velo_filtered);
  plane_segmentation.segment (*inliers, *coefficients);

  if (inliers->indices.size () == 0)
  {
    ROS_WARN("[Laser] Could not estimate a planar model for the given dataset.");
    return;
  }
  pcl::PointCloud<pcl::PointXYZ> inliers_filtered;
  uint32_t n_inliers = inliers->indices.size();

  inliers_filtered.points.resize (n_inliers);
  inliers_filtered.header   = velo_filtered->header;
  inliers_filtered.width    = n_inliers;
  inliers_filtered.height   = 1;
  inliers_filtered.is_dense = velo_filtered->is_dense;
  inliers_filtered.sensor_orientation_ = velo_filtered->sensor_orientation_;
  inliers_filtered.sensor_origin_ = velo_filtered->sensor_origin_;

  // Iterate over each point
  for (size_t i = 0; i < n_inliers; ++i) {
    const Velodyne::Point &pt = velo_filtered->points[inliers->indices[i]];
    inliers_filtered.points[i] = pcl::PointXYZ(pt.x,pt.y,pt.z);
  }
  sensor_msgs::PointCloud2 inliers_ros;
  pcl::toROSMsg(inliers_filtered, inliers_ros);
  inliers_ros.header = laser_cloud->header;
  inliers_pub.publish(inliers_ros);

  // Copy coefficients to proper object for further filtering
  Eigen::VectorXf coefficients_v(4);
  coefficients_v(0) = coefficients->values[0];
  coefficients_v(1) = coefficients->values[1];
  coefficients_v(2) = coefficients->values[2];
  coefficients_v(3) = coefficients->values[3];

  // Rotate cloud to face pattern plane
  pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Vector3d xy_plane_normal_vector, floor_plane_normal_vector;
  xy_plane_normal_vector[0] = 0.0;
  xy_plane_normal_vector[1] = 0.0;
  xy_plane_normal_vector[2] = -1.0;

  floor_plane_normal_vector[0] = coefficients->values[0];
  floor_plane_normal_vector[1] = coefficients->values[1];
  floor_plane_normal_vector[2] = coefficients->values[2];

  Eigen::Affine3d rotation = getRotationMatrix(floor_plane_normal_vector, xy_plane_normal_vector);
  pcl::transformPointCloud(inliers_filtered, *xy_cloud, rotation);

  pcl::PointCloud<pcl::PointXYZ>::Ptr aux_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointXYZ aux_point;
  aux_point.x = 0;
  aux_point.y = 0;
  aux_point.z = (-coefficients_v(3)/coefficients_v(2));
  aux_cloud->push_back(aux_point);

  pcl::PointCloud<pcl::PointXYZ>::Ptr auxrotated_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud(*aux_cloud, *auxrotated_cloud, rotation);

  sensor_msgs::PointCloud2 ros_auxpoint;
  pcl::toROSMsg(*xy_cloud, ros_auxpoint);
  ros_auxpoint.header = laser_cloud->header;
  auxpoint_pub.publish(ros_auxpoint);

  double zcoord_xyplane = auxrotated_cloud->at(0).z;

  pcl::PointXYZ inliers_centroid;
  pcl::computeCentroid(*xy_cloud,inliers_centroid);

  pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  copy_cloud->points.resize (n_inliers);
  copy_cloud->width    = n_inliers;
  copy_cloud->height   = 1;
  // Force pattern points to belong to computed plane
  for (size_t i = 0; i < n_inliers; ++i) {
    const pcl::PointXYZ &pt = xy_cloud->points[i];
    copy_cloud->points[i] = pcl::PointXYZ(pt.x,pt.y,zcoord_xyplane);
  }

  sensor_msgs::PointCloud2 ros_copycloud;
  pcl::toROSMsg(*copy_cloud, ros_copycloud);
  ros_copycloud.header = laser_cloud->header;
  debug_pub.publish(ros_copycloud);

  std::vector<pcl::PointXYZ> found_centers;

  if (detectCirclesInPointcloud(copy_cloud,found_centers)){
    for (auto center : found_centers){
      pcl::PointXYZ center_rotated_back = pcl::transformPoint(center, rotation.inverse());
      center_rotated_back.x = (- coefficients->values[1] * center_rotated_back.y - coefficients->values[2] * center_rotated_back.z - coefficients->values[3])/coefficients->values[0];
      cumulative_cloud->push_back(center_rotated_back);
    }

    sensor_msgs::PointCloud2 ros_pointcloud;
    pcl::toROSMsg(*cumulative_cloud, ros_pointcloud);
    ros_pointcloud.header = laser_cloud->header;
    cumulative_pub.publish(ros_pointcloud);
  }else{
    ROS_WARN("[Laser] Not enough centers: %ld", found_centers.size());
    return;
  }

  copy_cloud.reset(); // Free memory


  nFrames++;
  clouds_used_ = nFrames;

  pcl_msgs::ModelCoefficients m_coeff;
  pcl_conversions::moveFromPCL(*coefficients, m_coeff);
  m_coeff.header = laser_cloud->header;
  coeff_pub.publish(m_coeff);

  ROS_INFO("[Laser] %d/%d frames: %ld pts in cloud", clouds_used_, clouds_proc_, cumulative_cloud->points.size());

  // Create cloud for publishing centers
  pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Compute circles centers
  getCenterClusters(cumulative_cloud, centers_cloud, cluster_size_, nFrames/2, nFrames);
  if (centers_cloud->points.size()>4){
    getCenterClusters(cumulative_cloud, centers_cloud, cluster_size_, 3.0*nFrames/4.0, nFrames);
  }

  if (centers_cloud->points.size()==4){

    sensor_msgs::PointCloud2 ros2_pointcloud;
    pcl::toROSMsg(*centers_cloud, ros2_pointcloud);
    ros2_pointcloud.header = laser_cloud->header;

    velo2cam_calibration::ClusterCentroids to_send;
    to_send.header = laser_cloud->header;
    to_send.cluster_iterations = clouds_used_;
    to_send.total_iterations = clouds_proc_;
    to_send.cloud = ros2_pointcloud;

    centers_pub.publish(to_send);
    //if(DEBUG) ROS_INFO("Pattern centers published");
  }
  cumulative_cloud->clear();
}

void param_callback(velo2cam_calibration::LaserConfig &config, uint32_t level){
  passthrough_radius_min_ = config.passthrough_radius_min;
  ROS_INFO("[Laser] New minimum passthrough radius: %f", passthrough_radius_min_);
  passthrough_radius_max_ = config.passthrough_radius_max;
  ROS_INFO("[Laser] New maximum passthrough radius: %f", passthrough_radius_max_);
  circle_radius_ = config.circle_radius;
  ROS_INFO("[Laser] New pattern circle radius: %f", circle_radius_);
  circle_radius_threshold_ = config.circle_radius_threshold;
  ROS_INFO("[Laser] New pattern circle radius threshold: %f", circle_radius_threshold_);
  axis_[0] = config.x;
  axis_[1] = config.y;
  axis_[2] = config.z;
  ROS_INFO("[Laser] New normal axis for plane segmentation: %f, %f, %f", axis_[0], axis_[1], axis_[2]);
  angle_threshold_ = config.angle_threshold;
  ROS_INFO("[Laser] New angle threshold: %f", angle_threshold_);
  centroid_distance_min_ = config.centroid_distance_min;
  ROS_INFO("[Laser] New minimum distance between centroids: %f", centroid_distance_min_);
  centroid_distance_max_ = config.centroid_distance_max;
  ROS_INFO("[Laser] New maximum distance between centroids: %f", centroid_distance_max_);
  plane_distance_threshold_ = config.plane_distance_threshold;
  ROS_INFO("[Laser] New plane distance threshold: %f", plane_distance_threshold_);
  edge_range_threshold = config.edge_range_threshold;
  ROS_INFO("[Laser] New edge range threshold: %f", edge_range_threshold);
  min_distance_between_centers_ = config.min_distance_between_centers;
  ROS_INFO("[Laser] New minimum distance between centers: %f", min_distance_between_centers_);
}

int main(int argc, char **argv){
  ros::init(argc, argv, "laser_pattern");
  ros::NodeHandle nh_("~"); // LOCAL
  ros::Subscriber sub = nh_.subscribe ("cloud1", 1, callback);

  range_pub = nh_.advertise<PointCloud2> ("range_filtered_velo", 1);
  pattern_pub2 = nh_.advertise<PointCloud2> ("pattern_circles", 1);
  auxpoint_pub = nh_.advertise<PointCloud2> ("rotated_pattern", 1);
  cumulative_pub = nh_.advertise<PointCloud2> ("cumulative_cloud", 1);
  edges_cloud_pub = nh_.advertise<PointCloud2> ("edges_cloud", 1);
  inliers_pub = nh_.advertise<PointCloud2> ("inliers_cloud", 1);
  centers_pub = nh_.advertise<velo2cam_calibration::ClusterCentroids> ("centers_cloud", 1);
  pattern_pub = nh_.advertise<PointCloud2> ("pattern_cloud", 1);
  debug_pub = nh_.advertise<PointCloud2> ("debug", 1);

  coeff_pub = nh_.advertise<pcl_msgs::ModelCoefficients> ("plane_model", 1);

  nh_.param("cluster_size", cluster_size_, 0.02);
  nh_.param("min_centers_found", min_centers_found_, 4);
  nh_.param("n_rings", n_rings, 16);

  nFrames = 0;
  cumulative_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

  dynamic_reconfigure::Server<velo2cam_calibration::LaserConfig> server;
  dynamic_reconfigure::Server<velo2cam_calibration::LaserConfig>::CallbackType f;
  f = boost::bind(param_callback, _1, _2);
  server.setCallback(f);

  ros::spin();
  return 0;
}
