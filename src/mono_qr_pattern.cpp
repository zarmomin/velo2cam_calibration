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
  mono_qr_pattern: Find the circle centers in the color image
*/
#include <ros/ros.h>
#include "ros/package.h"
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <image_geometry/pinhole_camera_model.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>
#include <pcl/common/geometry.h>
#include <pcl/common/eigen.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iostream>

#include <dynamic_reconfigure/server.h>
#include <velo2cam_calibration/MonocularConfig.h>
#include "velo2cam_utils.h"
#include <velo2cam_calibration/ClusterCentroids.h>

using namespace std;
using namespace cv;

pcl::PointCloud<pcl::PointXYZ>::Ptr cumulative_cloud;
cv::Ptr<cv::aruco::Dictionary> dictionary;
ros::Publisher qr_pub, centers_cloud_pub, cumulative_pub, clusters_pub;

// ROS params
double marker_size_, delta_x_marker_circle_, delta_y_marker_circle_, delta_x_centers_, delta_y_centers_;

Eigen::Vector3f mean(pcl::PointCloud<pcl::PointXYZ>::Ptr cumulative_cloud){
  double x=0, y=0, z=0;
  int npoints = cumulative_cloud->points.size();
  for (pcl::PointCloud<pcl::PointXYZ>::iterator pt = cumulative_cloud->points.begin(); pt < cumulative_cloud->points.end(); pt++){
    x+=(pt->x)/npoints;
    y+=(pt->y)/npoints;
    z+=(pt->z)/npoints;
  }
  return Eigen::Vector3f(x,y,z);
}

Eigen::Matrix3f covariance(pcl::PointCloud<pcl::PointXYZ>::Ptr cumulative_cloud, Eigen::Vector3f means){
  double x=0, y=0, z=0;
  int npoints = cumulative_cloud->points.size();
  vector<Eigen::Vector3f> points;

  for (pcl::PointCloud<pcl::PointXYZ>::iterator pt = cumulative_cloud->points.begin(); pt < cumulative_cloud->points.end(); pt++){
    Eigen::Vector3f p(pt->x, pt->y, pt->z);
    points.push_back(p);
  }

  Eigen::Matrix3f covarianceMatrix(3, 3);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      covarianceMatrix(i, j) = 0.0;
      for (int k = 0; k < npoints; k++){
        covarianceMatrix(i, j) += (means[i] - points[k][i]) *
                                  (means[j] - points[k][j]);
      }
      covarianceMatrix(i, j) /= npoints - 1;
    }
  return covarianceMatrix;
}

void imageCallback(const sensor_msgs::ImageConstPtr& msg, const sensor_msgs::CameraInfoConstPtr& left_info) {
  cv_bridge::CvImageConstPtr cv_img_ptr;
  try {
    cv_img_ptr = cv_bridge::toCvShare(msg);
  } catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv::Mat image(cv_img_ptr->image.rows, cv_img_ptr->image.cols, cv_img_ptr->image.type());
  image = cv_bridge::toCvShare(msg)->image;
  cv::Mat imageCopy;
  image.copyTo(imageCopy);
  sensor_msgs::CameraInfoPtr cinfo(new sensor_msgs::CameraInfo(*left_info));
  image_geometry::PinholeCameraModel cam_model_;
  cam_model_.fromCameraInfo(cinfo);

  // TODO Not needed at each frame -> Move it to separate callback
  Mat cameraMatrix(3,3, CV_32F);
  cameraMatrix.at<float>(0, 0) = cinfo->K[0];
  cameraMatrix.at<float>(0, 1) = cinfo->K[1];
  cameraMatrix.at<float>(0, 2) = cinfo->K[2];
  cameraMatrix.at<float>(1, 0) = cinfo->K[3];
  cameraMatrix.at<float>(1, 1) = cinfo->K[4];
  cameraMatrix.at<float>(1, 2) = cinfo->K[5];
  cameraMatrix.at<float>(2, 0) = cinfo->K[6];
  cameraMatrix.at<float>(2, 1) = cinfo->K[7];
  cameraMatrix.at<float>(2, 2) = cinfo->K[8];

  Mat distCoeffs(1, cinfo->D.size(), CV_32F);
  for(int i=0; i<cinfo->D.size(); i++)
    distCoeffs.at<float>(0,i) = cinfo->D[i];

  // TODO End of block to move

  // Detect markers
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f> > corners;
  cv::aruco::detectMarkers(image, dictionary, corners, ids);

  pcl::PointCloud<pcl::PointXYZ>::Ptr qr_cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Draw detections if at least one marker detected
  if (ids.size() > 0)
    cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

  if(ids.size()==4){
    // Estimate 3D position of the markers
    vector< Vec3d > rvecs, tvecs;
    cv::aruco::estimatePoseSingleMarkers(corners, marker_size_, cameraMatrix, distCoeffs, rvecs, tvecs);

    // Draw markers' axis and centers in color image (Debug purposes)
    for(int i=0; i<ids.size(); i++){
      double x = tvecs[i][0];
      double y = tvecs[i][1];
      double z = tvecs[i][2];
      pcl::PointXYZ qr_center;
      qr_center.x = x;
      qr_center.y = y;
      qr_center.z = z;
      qr_cloud->push_back(qr_center);
      cv::Point3d pt_cv(x, y, z);
      cv::Point2d uv;
      uv = cam_model_.project3dToPixel(pt_cv);

      cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
      circle(imageCopy, uv, 12, Scalar(0,0,255), -1);
    }

    if(qr_cloud->points.size()>3){
      // Plane fitting
      Eigen::Vector3f mu = mean(qr_cloud);
      Eigen::Matrix3f covar = covariance(qr_cloud, mu);
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(covar, Eigen::ComputeFullU);
      Eigen::Vector3f normal = svd.matrixU().col(2);
      Eigen::Hyperplane<float, 3> result(normal, mu);

      // Rotate pattern plane to face XY plane so that we can work in 2D coords
      pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointCloud<pcl::PointXYZ>::Ptr centers_xy_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      Eigen::Vector3d xy_plane_normal_vector, floor_plane_normal_vector;
      xy_plane_normal_vector[0] = 0.0;
      xy_plane_normal_vector[1] = 0.0;
      xy_plane_normal_vector[2] = 1.0;

      floor_plane_normal_vector[0] = normal[0];
      floor_plane_normal_vector[1] = normal[1];
      floor_plane_normal_vector[2] = normal[2];

      Eigen::Affine3d rotation = getRotationMatrix(floor_plane_normal_vector, xy_plane_normal_vector);
      pcl::transformPointCloud(*qr_cloud, *xy_cloud, rotation.inverse());

      // Order vertices in 2D by comparing their relative position to their centroid
      vector<cv::Point2f> qr_centers;
      for(pcl::PointCloud<pcl::PointXYZ>::iterator it=xy_cloud->points.begin(); it<xy_cloud->points.end(); it++){
        qr_centers.push_back(cv::Point2f(it->x, it->y));
      }

      double avg_x = 0, avg_y = 0;
      for(vector<cv::Point2f>::iterator it=qr_centers.begin(); it<qr_centers.end(); it++){
        avg_x += (*it).x;
        avg_y += (*it).y;
      }

      cv::Point2f center;
      center.x = avg_x/4.;
      center.y = avg_y/4.;
      vector<cv::Point2f> v(4);
      for(vector<cv::Point2f>::iterator it=qr_centers.begin(); it<qr_centers.end(); it++){
        double x_dif = (*it).x - center.x;
        double y_dif = (*it).y - center.y;

        if(x_dif < 0 && y_dif < 0){
          v[0] = (*it);
        }else if(x_dif > 0 && y_dif < 0){
          v[1] = (*it);
        }else if(x_dif > 0 && y_dif > 0){
          v[2] = (*it);
        }else{
          v[3] = (*it);
        }
      }

      // Compute pattern XY angle and aux variables
      cv::Point2f upperEdge = v[1] - v[0];
      double ang = atan(upperEdge.y / upperEdge.x);
      double cosine = cos(ang);
      double sine = sin(ang);

      // Set keypoints
      double delta_x_marker_far_circle_ = delta_x_marker_circle_+ delta_x_centers_;
      double delta_y_marker_far_circle_ = delta_y_marker_circle_+ delta_y_centers_;

      // Compute centers coordinates (in 2D)
      pcl::PointXYZ circle1(v[0].x + (delta_x_marker_circle_ * cosine - delta_y_marker_circle_ * sine), v[0].y + (delta_y_marker_circle_ * cosine + delta_x_marker_circle_ * sine), xy_cloud->at(0).z);
      pcl::PointXYZ circle2(v[0].x + (delta_x_marker_far_circle_ * cosine - delta_y_marker_circle_ * sine), v[0].y + (delta_y_marker_circle_ * cosine + delta_x_marker_far_circle_ * sine), xy_cloud->at(1).z);
      pcl::PointXYZ circle3(v[0].x + (delta_x_marker_far_circle_ * cosine - delta_y_marker_far_circle_ * sine), v[0].y + (delta_y_marker_far_circle_ * cosine + delta_x_marker_far_circle_ * sine), xy_cloud->at(2).z);
      pcl::PointXYZ circle4(v[0].x + (delta_x_marker_circle_ * cosine - delta_y_marker_far_circle_ * sine), v[0].y + (delta_y_marker_far_circle_ * cosine + delta_x_marker_circle_ * sine), xy_cloud->at(3).z);

      centers_xy_cloud->push_back(circle1);
      centers_xy_cloud->push_back(circle2);
      centers_xy_cloud->push_back(circle3);
      centers_xy_cloud->push_back(circle4);

      // Rotate centers back to original 3D plane
      pcl::PointCloud<pcl::PointXYZ>::Ptr centers_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::transformPointCloud(*centers_xy_cloud, *centers_cloud, rotation);

      // Add centers to cumulative for further clustering
      cumulative_cloud->push_back(centers_cloud->at(0));
      cumulative_cloud->push_back(centers_cloud->at(1));
      cumulative_cloud->push_back(centers_cloud->at(2));
      cumulative_cloud->push_back(centers_cloud->at(3));

      // Draw centers
      cv::Point3d pt_circle1(centers_cloud->at(0).x, centers_cloud->at(0).y, centers_cloud->at(0).z);
      cv::Point2d uv_circle1;
      uv_circle1 = cam_model_.project3dToPixel(pt_circle1);
      circle(imageCopy, uv_circle1, 2, Scalar(255,0,255), -1);

      cv::Point3d pt_circle2(centers_cloud->at(1).x, centers_cloud->at(1).y, centers_cloud->at(1).z);
      cv::Point2d uv_circle2;
      uv_circle2 = cam_model_.project3dToPixel(pt_circle2);
      circle(imageCopy, uv_circle2, 2, Scalar(255,0,255), -1);

      cv::Point3d pt_circle3(centers_cloud->at(2).x, centers_cloud->at(2).y, centers_cloud->at(2).z);
      cv::Point2d uv_circle3;
      uv_circle3 = cam_model_.project3dToPixel(pt_circle3);
      circle(imageCopy, uv_circle3, 2, Scalar(255,0,255), -1);

      cv::Point3d pt_circle4(centers_cloud->at(3).x, centers_cloud->at(3).y, centers_cloud->at(3).z);
      cv::Point2d uv_circle4;
      uv_circle4 = cam_model_.project3dToPixel(pt_circle4);
      circle(imageCopy, uv_circle4, 2, Scalar(255,0,255), -1);

      // Compute centers clusters
      pcl::PointCloud<pcl::PointXYZ>::Ptr clusters_cloud(new pcl::PointCloud<pcl::PointXYZ>);
      getCenterClusters(cumulative_cloud, clusters_cloud, 0.02, 10, 10000);

      // Publish pointcloud messages
      sensor_msgs::PointCloud2 ros_pointcloud;
      pcl::toROSMsg(*qr_cloud, ros_pointcloud); //circles_cloud
      ros_pointcloud.header = msg->header;
      qr_pub.publish(ros_pointcloud);

      sensor_msgs::PointCloud2 centers_pointcloud;
      pcl::toROSMsg(*clusters_cloud, centers_pointcloud);
      centers_pointcloud.header = msg->header;
      centers_cloud_pub.publish(centers_pointcloud);

      velo2cam_calibration::ClusterCentroids to_send;
      to_send.header = msg->header;
      to_send.cluster_iterations = 0;
      to_send.total_iterations = 0;
      to_send.cloud = centers_pointcloud;
      clusters_pub.publish(to_send);

      sensor_msgs::PointCloud2 cumulative_pointcloud;
      pcl::toROSMsg(*cumulative_cloud, cumulative_pointcloud);
      cumulative_pointcloud.header = msg->header;
      cumulative_pub.publish(cumulative_pointcloud);
    }
  }
  cv::imshow("out", imageCopy);
  cv::waitKey(1);
}


void param_callback(velo2cam_calibration::MonocularConfig &config, uint32_t level){
  marker_size_ = config.marker_size;
  ROS_INFO("New marker_size_: %f", marker_size_);
  delta_x_marker_circle_ = config.delta_x_marker_circle;
  ROS_INFO("New delta_x_marker_circle_: %f", delta_x_marker_circle_);
  delta_y_marker_circle_ = config.delta_y_marker_circle;
  ROS_INFO("New delta_y_marker_circle_: %f", delta_y_marker_circle_);
  delta_x_centers_ = config.delta_x_centers;
  ROS_INFO("New delta_x_centers_: %f", delta_x_centers_);
  delta_y_centers_ = config.delta_y_centers;
  ROS_INFO("New delta_y_centers_: %f", delta_y_centers_);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "mono_qr_pattern");
  ros::NodeHandle nh_("~");

  // Initialize QR dictionary
  dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

  cumulative_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
  qr_pub = nh_.advertise<sensor_msgs::PointCloud2> ("qr_cloud", 1);
  centers_cloud_pub = nh_.advertise<sensor_msgs::PointCloud2> ("centers_pts_cloud", 1);
  cumulative_pub = nh_.advertise<sensor_msgs::PointCloud2> ("cumulative_cloud", 1);
  clusters_pub = nh_.advertise<velo2cam_calibration::ClusterCentroids> ("centers_cloud", 1);

  nh_.param("marker_size", marker_size_, 0.20);
  nh_.param("delta_x_marker_circle", delta_x_marker_circle_, 0.30);
  nh_.param("delta_y_marker_circle", delta_y_marker_circle_, 0.10);
  nh_.param("delta_x_centers", delta_x_centers_, 0.30);
  nh_.param("delta_y_centers", delta_y_centers_, 0.30);

  message_filters::Subscriber<sensor_msgs::Image> image_sub(nh_, "/stereo_camera/left/image_rect_color", 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo> cinfo_sub(nh_, "/stereo_camera/left/camera_info", 1);

  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_sub, cinfo_sub, 10);
  sync.registerCallback(boost::bind(&imageCallback, _1, _2));

  // ROS param callback
  dynamic_reconfigure::Server<velo2cam_calibration::MonocularConfig> server;
  dynamic_reconfigure::Server<velo2cam_calibration::MonocularConfig>::CallbackType f;
  f = boost::bind(param_callback, _1, _2);
  server.setCallback(f);

  ros::spin();
  cv::destroyAllWindows();
  return 0;
}

