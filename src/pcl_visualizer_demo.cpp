

#include <iostream>

#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr(new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

// coordinate frame transform
float x, y, z;
Eigen::Affine3f transform;
float angle_in_radian = 3.14 / 32.0;
float translate_in_m = 0.002;
float coordinate_frame_scale = 0.2;

// --------------
// -----Help-----
// --------------

void
printUsage(const char* progName) {
  std::cout << "\n\nUsage: " << progName << " [options]\n\n"
          << "Options:\n"
          << "-------------------------------------------\n"
          << "-h           this help\n"
          << "\n\n";
}

int text_id = 0;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void) {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);

  //  std::cout << "key pressed: " << event.getKeySym() << std::endl;
  Eigen::AngleAxis<float> aa;
  if (event.getKeySym() == "r" && event.keyDown()) {
    viewer->removeCoordinateSystem();
    return;
  } else if (event.getKeySym() == "a" && event.keyDown()) {
    // +x rotation
    aa = Eigen::AngleAxis<float>(angle_in_radian, Eigen::Vector3f(1, 0, 0));
    transform.rotate(aa);
  } else if (event.getKeySym() == "s" && event.keyDown()) {
    // +y rotation
    aa = Eigen::AngleAxis<float>(angle_in_radian, Eigen::Vector3f(0, 1, 0));
    transform.rotate(aa);
  } else if (event.getKeySym() == "d" && event.keyDown()) {
    // +z rotation
    aa = Eigen::AngleAxis<float>(angle_in_radian, Eigen::Vector3f(0, 0, 1));
    transform.rotate(aa);
  } else if (event.getKeySym() == "A" && event.keyDown()) {
    // -x rotation
    aa = Eigen::AngleAxis<float>(-angle_in_radian, Eigen::Vector3f(1, 0, 0));
    transform.rotate(aa);
  } else if (event.getKeySym() == "S" && event.keyDown()) {
    // -y rotation
    aa = Eigen::AngleAxis<float>(-angle_in_radian, Eigen::Vector3f(0, 1, 0));
    transform.rotate(aa);
  } else if (event.getKeySym() == "D" && event.keyDown()) {
    // -z rotation
    aa = Eigen::AngleAxis<float>(-angle_in_radian, Eigen::Vector3f(0, 0, 1));
    transform.rotate(aa);

  } else if (event.getKeySym() == "f" && event.keyDown()) {
    // +x translation
    transform.translate(Eigen::Vector3f(translate_in_m, 0, 0));
  } else if (event.getKeySym() == "g" && event.keyDown()) {
    // +y translation
    transform.translate(Eigen::Vector3f(0, translate_in_m, 0));
  } else if (event.getKeySym() == "h" && event.keyDown()) {
    // +z translation
    transform.translate(Eigen::Vector3f(0, 0, translate_in_m));
  } else if (event.getKeySym() == "F" && event.keyDown()) {
    // -x translation
    transform.translate(Eigen::Vector3f(-translate_in_m, 0, 0));
  } else if (event.getKeySym() == "G" && event.keyDown()) {
    // -y translation
    transform.translate(Eigen::Vector3f(0, -translate_in_m, 0));
  } else if (event.getKeySym() == "H" && event.keyDown()) {
    // -z translation
    transform.translate(Eigen::Vector3f(0, 0, -translate_in_m));

  } else if (event.getKeySym() == "p" && event.keyDown()) {
    // print rotation and translation

    Eigen::Matrix3f rotation = transform.rotation();
    Eigen::Quaternionf quat(rotation);
    Eigen::Vector3f translation = transform.translation();

    printf("<point>\n");
    printf("<x>%0.3f</x>\n", translation(0));
    printf("<y>%0.3f</y>\n", translation(1));
    printf("<z>%0.3f</z>\n", translation(2));
    printf("</point>\n");
    printf("<orientation>\n");
    printf("<x>%0.3f</x>\n", quat.x());
    printf("<y>%0.3f</y>\n", quat.y());
    printf("<z>%0.3f</z>\n", quat.z());
    printf("<w>%0.3f</w>\n", quat.w());
    printf("</orientation>\n");

  } else {
    return;
  }


  // perform rotation or translation
  viewer->removeCoordinateSystem();
  viewer->addCoordinateSystem(coordinate_frame_scale, transform);
}

void mouseEventOccurred(const pcl::visualization::MouseEvent &event, void* viewer_void) {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *> (viewer_void);
  if (event.getButton() == pcl::visualization::MouseEvent::LeftButton &&
          event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease) {
    std::cout << "Left mouse button released at position (" << event.getX() << ", " << event.getY() << ")" << std::endl;

    char str[512];
    sprintf(str, "text#%03d", text_id++);
    viewer->addText("clicked here", event.getX(), event.getY(), str);
  }
}

void pp_callback(const pcl::visualization::PointPickingEvent& event, void* viewer_void) {
  std::cout << "In pp_callback.\n";

  int idx = event.getPointIndex();
  if (idx == -1) {
    std::cout << "index is -1. returning.\n";
    return;
  }

  event.getPoint(x, y, z);
  //  PCL_INFO("Point index picked: %d - [%f, %f, %f]\n", idx, x, y, z);

  Eigen::Affine3f r(Eigen::Quaternionf(1, 0, 0, 0)); //w, x, y, z
  Eigen::Affine3f t(Eigen::Translation3f(x, y, z));
  transform = (t * r).matrix();

  viewer->addCoordinateSystem(coordinate_frame_scale, transform);

}

boost::shared_ptr<pcl::visualization::PCLVisualizer> initVis() {
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->setBackgroundColor(0, 0, 0);
  viewer->addCoordinateSystem(coordinate_frame_scale);

  viewer->registerKeyboardCallback(keyboardEventOccurred, (void*) &viewer);
  viewer->registerPointPickingCallback(pp_callback, (void*) &(*viewer));

  return (viewer);
}


// --------------
// -----Main-----
// --------------

int
main(int argc, char** argv) {
  // --------------------------------------
  // -----Parse Command Line Arguments-----
  // --------------------------------------
  if (pcl::console::find_argument(argc, argv, "-h") >= 0) {
    printUsage(argv[0]);
    return 0;
  }

  std::string cluster_filename = argv[1];
  pcl::console::print_highlight("Loading input cluster: %s\n", cluster_filename.c_str());

  if (pcl::io::loadPCDFile(cluster_filename, *point_cloud_ptr) < 0) {
    std::cout << "Error loading model cloud: " << cluster_filename << std::endl;
    return false;
  }

  // ------------------------------------
  // -----Create example point cloud-----
  // ------------------------------------
  if (point_cloud_ptr->points.size() == 0) {
    std::cout << "Genarating example point clouds.\n\n";
    // We're going to make an ellipse extruded along the z-axis. The colour for
    // the XYZRGB cloud will gradually go from red to green to blue.
    uint8_t r(255), g(15), b(15);
    for (float z(-1.0); z <= 1.0; z += 0.05) {
      for (float angle(0.0); angle <= 360.0; angle += 5.0) {
        pcl::PointXYZ basic_point;
        basic_point.x = 0.5 * cosf(pcl::deg2rad(angle));
        basic_point.y = sinf(pcl::deg2rad(angle));
        basic_point.z = z;
        basic_cloud_ptr->points.push_back(basic_point);

        pcl::PointXYZRGB point;
        point.x = basic_point.x;
        point.y = basic_point.y;
        point.z = basic_point.z;
        uint32_t rgb = (static_cast<uint32_t> (r) << 16 |
                static_cast<uint32_t> (g) << 8 | static_cast<uint32_t> (b));
        point.rgb = *reinterpret_cast<float*> (&rgb);
        point_cloud_ptr->points.push_back(point);
      }
      if (z < 0.0) {
        r -= 12;
        g += 12;
      } else {
        g -= 12;
        b += 12;
      }
    }
    basic_cloud_ptr->width = (int) basic_cloud_ptr->points.size();
    basic_cloud_ptr->height = 1;
    point_cloud_ptr->width = (int) point_cloud_ptr->points.size();
    point_cloud_ptr->height = 1;
  }

  //  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = initVis();
  viewer = initVis();

  //add point cloud
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> cloud_color_handler(point_cloud_ptr, 0, 255, 0);
  viewer->addPointCloud(point_cloud_ptr, cloud_color_handler, "cloud");

  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    boost::this_thread::sleep(boost::posix_time::microseconds(100000));
  }
}