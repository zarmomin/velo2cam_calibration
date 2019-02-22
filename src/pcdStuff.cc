//
// Created by nico on 22.02.19.
//
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/common.h>

bool detectCirclesInPointcloud(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::PointXYZ>& centers)
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
  cv::HoughCircles(closed, circles, cv::HOUGH_GRADIENT, 1, cvRound(0.16*scale), 150, 10, cvRound(0.06*scale), cvRound(0.1*scale));
  /*std::cout << "\nCircles detected\n";
  cv::Mat img;
  cv::cvtColor(inliner_plane,img, cv::COLOR_GRAY2BGR);
  std::cout << "\nColorization done\n";*/
  for( size_t i = 0; i < circles.size(); i++ )
  {
    cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
    /*int radius = cvRound(circles[i][2]);
    // draw the circle center
    circle( img, center, 3, cv::Scalar(0,255,0), -1, 8, 0 );
    // draw the circle outline
    circle( img, center, radius, cv::Scalar(0,0,255), 3, 8, 0 );*/

    double x = center.x / scale + min_pt.x;
    double y = center.y / scale + min_pt.y;
    // all z values are identical anyways
    centers.push_back(pcl::PointXYZ(x, y, max_pt.z));
  }
  /*cv::Mat edggy, detected_edges;
  cv::Canny(closed, detected_edges, 100, 150);
  closed.copyTo(edggy, detected_edges);
  cv::imshow("edges", edggy);
  cv::imshow( "circles", img );
  cv::waitKey(0);*/
  return centers.size() == 4;
}

int main(int argc, char **argv)
{
  if (argc > 1)
  {
    std::string filename(argv[1]);
    pcl::PointCloud<pcl::PointXYZ>::Ptr points(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ> centers;
    pcl::io::loadPCDFile(filename, *points);
    detectCirclesInPointcloud(points, centers);
  }
}