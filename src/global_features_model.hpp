#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/features/crh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/board.h>
#include <pcl/common/transforms.h>

template <typename PointType, typename NormalType, typename DescriptorType>
class ObjectModel {
  typedef pcl::Histogram<90> CRH90;
  
public:

  ObjectModel(const std::string& type)
  : type_(type),
  cloud_(new pcl::PointCloud<PointType> ()),
  normals_(new pcl::PointCloud<NormalType> ()),
  descriptors_(new pcl::PointCloud<DescriptorType> ()),
  crh_(new pcl::PointCloud<CRH90>),
  centroid_() {
  }

  ~ObjectModel() {

  }

  std::string type_;
  typename pcl::PointCloud<PointType>::Ptr cloud_;
  typename pcl::PointCloud<NormalType>::Ptr normals_;
  typename pcl::PointCloud<DescriptorType>::Ptr descriptors_;
  pcl::PointCloud<CRH90>::Ptr crh_;
  Eigen::Vector4f centroid_;
};