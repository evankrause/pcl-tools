#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/vfh.h>
#include <pcl/features/crh.h>
#include <pcl/features/cvfh.h>
#include <pcl/features/our_cvfh.h>
#include <pcl/features/board.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/recognition/crh_alignment.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/surface/mls.h>

#include "global_features_model.hpp"

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::VFHSignature308 DescriptorType;

std::string model_dir_;
std::string cluster_filename_;

//Algorithm params
bool show_matches_(false);
int feature_type_(0);

void
showHelp(char *filename) {
  std::cout << std::endl;
  std::cout << "***************************************************************************" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "*             Correspondence Grouping Tutorial - Usage Guide              *" << std::endl;
  std::cout << "*                                                                         *" << std::endl;
  std::cout << "***************************************************************************" << std::endl << std::endl;
  std::cout << "Usage: " << filename << " model_dir cluster_filename.pcd [Options]" << std::endl << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     -h:                          Show this help." << std::endl;
  std::cout << "     -m:                          Show found matches." << std::endl;
  std::cout << "     --ft [VFH | CVFH | OUCVFH]:  Kind of feature to use." << std::endl;
}

void
parseCommandLine(int argc, char *argv[]) {
  //Show help
  if (pcl::console::find_switch(argc, argv, "-h")) {
    showHelp(argv[0]);
    exit(0);
  }

  model_dir_ = argv[1];
  cluster_filename_ = argv[2];
  pcl::console::print_highlight("Loading model directory: %s Loading input cluster: %s\n", model_dir_.c_str(), cluster_filename_.c_str());

  //Program behavior
  if (pcl::console::find_switch(argc, argv, "-m")) {
    show_matches_ = true;
  }

  std::string used_feature;
  if (pcl::console::parse_argument(argc, argv, "--ft", used_feature) != -1) {
    if (used_feature.compare("VFH") == 0) {
      feature_type_ = 0;
    } else if (used_feature.compare("CVFH") == 0) {
      feature_type_ = 1;
    } else if (used_feature.compare("OURCVFH") == 0) {
      feature_type_ = 2;
    } else {
      std::cout << "Wrong feature type.\n";
      showHelp(argv[0]);
      exit(-1);
    }
  }

  //General parameters
  //  pcl::console::parse_argument(argc, argv, "--model_ss", model_ss_);
  //  pcl::console::parse_argument(argc, argv, "--scene_ss", scene_ss_);
  //  pcl::console::parse_argument(argc, argv, "--rf_rad", rf_rad_);
  //  pcl::console::parse_argument(argc, argv, "--descr_rad", descr_rad_);
  //  pcl::console::parse_argument(argc, argv, "--cg_size", cg_size_);
  //  pcl::console::parse_argument(argc, argv, "--cg_thresh", cg_thresh_);
}

void printMatrix4f(Eigen::Matrix4f matrix, std::string matrix_name) {
  // Print the rotation matrix and translation vector
  Eigen::Matrix3f rotation = matrix.block<3, 3>(0, 0);
  Eigen::Vector3f translation = matrix.block<3, 1>(0, 3);

  printf("%s: \n", matrix_name.c_str());
  printf("            | %6.3f %6.3f %6.3f | \n", rotation(0, 0), rotation(0, 1), rotation(0, 2));
  printf("        R = | %6.3f %6.3f %6.3f | \n", rotation(1, 0), rotation(1, 1), rotation(1, 2));
  printf("            | %6.3f %6.3f %6.3f | \n", rotation(2, 0), rotation(2, 1), rotation(2, 2));
  printf("\n");
  printf("        t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));
}

void transRotToPose(const Eigen::Vector4f& translation, const Eigen::Quaternionf& orientation, Eigen::Matrix4f& pose) {
  Eigen::Affine3f r(orientation);
  Eigen::Affine3f t(Eigen::Translation3f(translation(0), translation(1), translation(2)));
  printf("q = < %0.3f, %0.3f, %0.3f, %0.3f >\n", orientation.x(), orientation.y(), orientation.z(), orientation.w());
  printf("t = < %0.3f, %0.3f, %0.3f >\n", translation(0), translation(1), translation(2));

  pose = (t * r).matrix();
  printMatrix4f(pose, "pose");
}

float computeMeshResolution(pcl::PointCloud<PointType>::ConstPtr& input) {
  typedef typename pcl::KdTree<PointType>::Ptr KdTreeInPtr;
  KdTreeInPtr tree = boost::make_shared<pcl::KdTreeFLANN<PointType> > (false);
  tree->setInputCloud(input);

  std::vector<int> nn_indices(9);
  std::vector<float> nn_distances(9);
  std::vector<int> src_indices;

  float sum_distances = 0.0;
  std::vector<float> avg_distances(input->points.size());
  // Iterate through the source data set
  for (size_t i = 0; i < input->points.size(); ++i) {
    tree->nearestKSearch(input->points[i], 9, nn_indices, nn_distances);

    float avg_dist_neighbours = 0.0;
    for (size_t j = 1; j < nn_indices.size(); j++)
      avg_dist_neighbours += sqrtf(nn_distances[j]);

    avg_dist_neighbours /= static_cast<float> (nn_indices.size());

    avg_distances[i] = avg_dist_neighbours;
    sum_distances += avg_dist_neighbours;
  }

  std::sort(avg_distances.begin(), avg_distances.end());
  float avg = avg_distances[static_cast<int> (avg_distances.size()) / 2 + 1];
  return avg;
}

void makeUniformResolution(pcl::PointCloud<PointType>::ConstPtr cloud,
        pcl::PointCloud<PointType>::Ptr& cloud_resampled) {

  printf("input cloud points: %lu, isOrganized: %s.\n", cloud->points.size(), (cloud->isOrganized() ? "true" : "false"));
//  pcl::visualization::PCLVisualizer viewer("MLS");
//  viewer.addPointCloud(cloud, "cloud");
//  viewer.spin();

  //
  // Moving Least Squares (MLS) Upsampling
  //

  pcl::PointCloud<PointType>::Ptr out(new pcl::PointCloud<PointType> ());

  //  pcl::MovingLeastSquares<PointType, PointType> mls;
  //  typename pcl::search::KdTree<PointType>::Ptr tree;
  //  Eigen::Vector4f centroid_cluster;
  //  pcl::compute3DCentroid(*cloud, centroid_cluster);
  //  float dist_to_sensor = centroid_cluster.norm();
  //  float sigma = dist_to_sensor * 0.01f;
  //  mls.setInputCloud(cloud);
  //  mls.setSearchMethod(tree);
  //  mls.setSearchRadius(sigma);
  //  mls.setUpsamplingMethod(mls.SAMPLE_LOCAL_PLANE);
  //  mls.setUpsamplingRadius(0.005);//(0.002);
  //  mls.setUpsamplingStepSize(0.001);
  //
  //  mls.process(*out);
  //  
  //  printf("mls cloud points: %lu, isOrganized: %s.\n", out->points.size(), (out->isOrganized() ? "true" : "false"));
  //  if (out->points.size() == 0) {
  //    PCL_WARN("NORMAL estimator: Cloud has no points after mls, wont be able to compute normals!\n");
  //    return;
  //  }
  //  viewer.removeAllPointClouds();
  //  viewer.addPointCloud(out, "cloud");
  //  viewer.spin();

  //
  // Downsample to set resolution
  //

  bool compute_mesh_resolution_ = true;
  float grid_resolution_ = 0.003f;
  float factor_voxel_grid_ = 2.0f;//3.0f;
  float voxel_grid_size = grid_resolution_;
  if (compute_mesh_resolution_) {
    voxel_grid_size = computeMeshResolution(cloud) * factor_voxel_grid_;
  }

  pcl::PointCloud<PointType>::Ptr out2(new pcl::PointCloud<PointType> ());
  pcl::VoxelGrid<PointType> grid;
  grid.setInputCloud(cloud);
  //  grid.setInputCloud(out);
  grid.setLeafSize(voxel_grid_size, voxel_grid_size, voxel_grid_size);
  grid.setDownsampleAllData(true);
  grid.filter(*out2);
  out = out2;

  printf("output cloud points: %lu, isOrganized: %s.\n", out->points.size(), (out->isOrganized() ? "true" : "false"));
  if (out->points.size() == 0) {
    PCL_WARN("NORMAL estimator: Cloud has no points after voxel grid, wont be able to compute normals!\n");
    return;
  }
//  viewer.removeAllPointClouds();
//  viewer.addPointCloud(out2, "cloud");
//  viewer.spin();

  cloud_resampled = out;

}

void computeVFHDescriptors(pcl::PointCloud<PointType>::ConstPtr cloud,
        pcl::PointCloud<NormalType>::ConstPtr normals,
        pcl::PointCloud<DescriptorType>::Ptr descriptors) {

  // Create the VFH estimation class, and pass the input dataset+normals to it

  pcl::VFHEstimation<PointType, NormalType, DescriptorType> vfh;
  vfh.setInputCloud(cloud);
  vfh.setInputNormals(normals);

  // Create an empty kdtree representation, and pass it to the FPFH estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType> ());
  vfh.setSearchMethod(tree);

  // Output datasets
  //pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

  // Compute the features
  vfh.compute(*descriptors);

  // vfhs->points.size () should be of size 1*
}

//void computeOURCVFHDescriptors(pcl::PointCloud<PointType>::ConstPtr cloud,
//        pcl::PointCloud<NormalType>::ConstPtr normals,
//        pcl::PointCloud<DescriptorType>::Ptr descriptors) {
//
//  pcl::OURCVFHEstimation<PointType, NormalType, DescriptorType> ourcvfh;
//  pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
//  ourcvfh.setSearchMethod(kdtree);
//  ourcvfh.setEPSAngleThreshold(5.0 / 180.0 * M_PI); // 5 degrees.
//  ourcvfh.setCurvatureThreshold(1.0);
//  ourcvfh.setNormalizeBins(false);
//  // Set the minimum axis ratio between the SGURF axes. At the disambiguation phase,
//  // this will decide if additional Reference Frames need to be created, if ambiguous.
//  ourcvfh.setAxisRatio(0.8);
//
//  ourcvfh.setInputCloud(cloud);
//  ourcvfh.setInputNormals(normals);
//  ourcvfh.compute(*descriptors);
//}

void computeCRH(pcl::PointCloud<PointType>::ConstPtr cloud,
        pcl::PointCloud<NormalType>::ConstPtr normals,
        Eigen::Vector4f& centroid,
        pcl::PointCloud<pcl::Histogram<90> >::Ptr histogram) {

  // CRH estimation object.

  pcl::CRHEstimation<PointType, NormalType, pcl::Histogram<90> > crh;
  crh.setInputCloud(cloud);
  crh.setInputNormals(normals);
  pcl::compute3DCentroid(*cloud, centroid);
  crh.setCentroid(centroid);

  // Compute the CRH.
  crh.compute(*histogram);
}

void estimatePose(pcl::PointCloud<PointType>::Ptr model,
        pcl::PointCloud<PointType>::Ptr scene,
        const Eigen::Vector4f& model_centroid,
        const Eigen::Vector4f& scene_centroid,
        pcl::PointCloud<pcl::Histogram<90> >::Ptr model_crh,
        pcl::PointCloud<pcl::Histogram<90> >::Ptr scene_crh,
        std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> >& poses) {

  // CRHAlignment works with Vector3f, not Vector4f.

  Eigen::Vector3f model_centroid3f(model_centroid(0), model_centroid(1), model_centroid(2));
  Eigen::Vector3f scene_centroid3f(scene_centroid(0), scene_centroid(1), scene_centroid(2));

  pcl::CRHAlignment<PointType, 90> alignment;
  //alignment.setInputAndTargetView(model, scene);
  alignment.setInputAndTargetCentroids(model_centroid3f, scene_centroid3f);
  alignment.align(*model_crh, *scene_crh);

  std::vector < Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > roll_transforms;
  alignment.getTransforms(roll_transforms);
  //std::cout << "Number of roll transforms: " << roll_transforms.size() << std::endl;
  //printMatrix4f(roll_transforms[0], "roll_transforms[0]");

  Eigen::Matrix4f model_pose;
  transRotToPose(model->sensor_origin_, model->sensor_orientation_, model_pose);

  Eigen::Matrix4f pose_transform;
  pose_transform = Eigen::Matrix4f(roll_transforms[0] * model_pose);
  poses.push_back(pose_transform);

  // show CRHs
  //  pcl::visualization::PCLHistogramVisualizer crh_visualizer;
  //  crh_visualizer.addFeatureHistogram<pcl::Histogram<90> >(*model_crh, 90, "model feature");
  //  crh_visualizer.addFeatureHistogram<pcl::Histogram<90> >(*scene_crh, 90, "scene feature");
  //  crh_visualizer.spin();
}

bool createModel(const boost::filesystem::path& model_file, ObjectModel<PointType, NormalType, DescriptorType>& model) {
  pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
  if (pcl::io::loadPCDFile(model_file.string(), *cloud) < 0) {
    std::cout << "Error loading model cloud: " << model_file << std::endl;
    return false;
  }

  model.type_ = model_file.string();

  std::cout << "1" << std::endl;
  //
  // make uniform resolution
  //
  makeUniformResolution(cloud, model.cloud_);

  std::cout << "2" << std::endl;
  //
  //  Compute Normals
  //
  //  pcl::NormalEstimationOMP<PointType, NormalType> norm_est;
  pcl::NormalEstimation<PointType, NormalType> norm_est;
  norm_est.setKSearch(10);
  norm_est.setInputCloud(model.cloud_);
  norm_est.compute(*(model.normals_));

  std::cout << "3" << std::endl;
  //
  //  Compute Descriptor
  //
  if (feature_type_ == 0) {
    computeVFHDescriptors(model.cloud_, model.normals_, model.descriptors_);
  } else if (feature_type_ == 1) {
    //  computeOURCVFHDescriptors(model, model_normals, model_descriptors);
  } else if (feature_type_ == 2) {

  } else {
    std::cout << "Trying to use unknown feature type." << std::endl;

    return false;
  }

  std::cout << "4" << std::endl;
  //
  // Compute Camera Roll Histogram
  //
  computeCRH(model.cloud_, model.normals_, model.centroid_, model.crh_);

  std::cout << "5" << std::endl;
  return true;
}

void loadFeatureModels(const boost::filesystem::path &base_dir,
        const std::string &extension,
        std::vector<ObjectModel<PointType, NormalType, DescriptorType> >& models,
        const std::string& test_model = "") {

  if (!boost::filesystem::exists(base_dir) || !boost::filesystem::is_directory(base_dir)) {
    pcl::console::print_highlight("Problem loading models from directory: %s\n", base_dir.c_str());
    return;
  }

  for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator(); ++it) {
    if (boost::filesystem::is_directory(it->status())) {
      std::stringstream ss;
      ss << it->path();
      pcl::console::print_highlight("Loading %s (%lu models found so far).\n", ss.str().c_str(), (unsigned long) models.size());
      loadFeatureModels(it->path(), extension, models, test_model);
    } else if (boost::filesystem::is_regular_file(it->status()) && boost::filesystem::extension(it->path()) == extension) {
      ObjectModel<PointType, NormalType, DescriptorType> model("TODO");
      boost::filesystem::path path(base_dir / it->path().filename());

      if (test_model.compare(path.string()) == 0) {
        pcl::console::print_highlight("Skippping test model: %s\n", path.string().c_str());
        continue;
      }

      if (createModel(base_dir / it->path().filename(), model)) {
        models.push_back(model);
      }
    } else {

      pcl::console::print_highlight("Problem loading model or directory: %s\n", it->path().string().c_str());
    }
  }
}

int
main(int argc, char *argv[]) {
  parseCommandLine(argc, argv);

  //
  //  Load clouds
  //
  std::vector<ObjectModel<PointType, NormalType, DescriptorType> > models;
  ObjectModel<PointType, NormalType, DescriptorType> cluster("unknown");

  // load models
  loadFeatureModels(model_dir_, ".pcd", models, cluster_filename_);
  pcl::console::print_highlight("Loaded %lu models.\n", (unsigned long) models.size());

  // load input cluster (that we're trying to classify)
  boost::filesystem::path cluster_path(cluster_filename_);
  createModel(cluster_path, cluster);

  // build mapping from descriptor index to model index (bc single model can have multiple descriptors)
  pcl::PointCloud<DescriptorType>::Ptr model_descriptors(new pcl::PointCloud<DescriptorType>());
  std::map<int, int> descriptor_to_model;
  std::vector<ObjectModel<PointType, NormalType, DescriptorType> >::const_iterator model_itr;
  pcl::PointCloud<DescriptorType>::const_iterator descriptor_iter;
  int model_idx = 0;
  int descriptor_idx = 0;
  for (model_itr = models.begin(); model_itr != models.end(); ++model_itr) {
    for (descriptor_iter = model_itr->descriptors_->begin(); descriptor_iter != model_itr->descriptors_->end(); ++descriptor_iter) {
      model_descriptors->push_back(*descriptor_iter);
      descriptor_to_model[descriptor_idx] = model_idx;
      ++descriptor_idx;
    }
    ++model_idx;
  }


  pcl::visualization::PCLHistogramVisualizer vfh_visualizer;
  vfh_visualizer.addFeatureHistogram<pcl::VFHSignature308 >(*(cluster.descriptors_), 308, "model feature");
  vfh_visualizer.spin();

  //
  //  Matching: Find Model-Scene Correspondences with KdTree
  //  
  pcl::CorrespondencesPtr model_scene_corrs(new pcl::Correspondences());
  {
    pcl::ScopeTime t("NN Matching");

    pcl::KdTreeFLANN<DescriptorType> match_search;
    match_search.setInputCloud(model_descriptors);

    //  For each scene descriptor, find nearest neighbor into the model descriptor cloud and add it to the correspondences vector.
    for (size_t i = 0; i < cluster.descriptors_->size(); ++i) {
      int num_neighs = 1; //TODO: update code below to handle +1 neighbors
      std::vector<int> neigh_indices(num_neighs);
      std::vector<float> neigh_sqr_dists(num_neighs);
      if (!pcl_isfinite(cluster.descriptors_->at(i).histogram[0])) //skipping NaNs
      {
        continue;
      }

      int found_neighs = match_search.nearestKSearch(cluster.descriptors_->at(i), num_neighs, neigh_indices, neigh_sqr_dists);
      std::cout << "neigh_sqr_dists[0]: " << neigh_sqr_dists[0] << std::endl;
      if (found_neighs == num_neighs) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
        //      if (found_neighs == num_neighs && neigh_sqr_dists[0] < 0.35f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
      {
        pcl::Correspondence corr(static_cast<int> (i), neigh_indices[0], neigh_sqr_dists[0]);
        model_scene_corrs->push_back(corr);
      }
    }
    std::cout << "Correspondences found: " << model_scene_corrs->size() << std::endl;
  }

  //
  // Pose estimation
  //

  std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms;
  ObjectModel<PointType, NormalType, DescriptorType> matching_model("unknown");
  {
    pcl::ScopeTime t("Pose estimation");

    //get matching model
    int model_idx = descriptor_to_model[model_scene_corrs->at(0).index_match];
    matching_model = models[model_idx];
    std::cout << "matching model: " << matching_model.type_ << std::endl;

    estimatePose(matching_model.cloud_, cluster.cloud_, matching_model.centroid_, cluster.centroid_, matching_model.crh_, cluster.crh_, transforms);
  }


  //
  // ICP pose refinement
  //
  std::vector<pcl::PointCloud<PointType>::ConstPtr> model_clouds_aligned;
  if (false) {
    pcl::ScopeTime t("Pose refinement");

    //Prepare scene and model clouds for the pose refinement step
    float VOXEL_SIZE_ICP_ = 0.005f;
    int ICP_iterations_ = 5;
    pcl::PointCloud<PointType>::Ptr model_voxelized_icp(new pcl::PointCloud<PointType> ());
    pcl::PointCloud<PointType>::Ptr cluster_voxelized_icp(new pcl::PointCloud<PointType> ());
    pcl::VoxelGrid<PointType> voxel_grid_icp;
    voxel_grid_icp.setLeafSize(VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_, VOXEL_SIZE_ICP_);

    voxel_grid_icp.setInputCloud(matching_model.cloud_);
    voxel_grid_icp.filter(*model_voxelized_icp);
    voxel_grid_icp.setInputCloud(cluster.cloud_);
    voxel_grid_icp.filter(*cluster_voxelized_icp);


    pcl::PointCloud<PointType>::Ptr model_aligned(new pcl::PointCloud<PointType>);
    pcl::transformPointCloud(*model_voxelized_icp, *model_aligned, transforms[0]);

    pcl::IterativeClosestPoint<PointType, PointType> reg;
    reg.setInputSource(model_aligned); //model
    reg.setInputTarget(cluster_voxelized_icp); //scene
    reg.setMaximumIterations(ICP_iterations_);
    reg.setMaxCorrespondenceDistance(VOXEL_SIZE_ICP_ * 3.f);
    reg.setTransformationEpsilon(1e-5);

    pcl::PointCloud<PointType>::Ptr output_(new pcl::PointCloud<PointType> ());
    reg.align(*output_);
    model_clouds_aligned.push_back(output_);

    if (reg.hasConverged()) {
      Eigen::Matrix4f icp_trans = reg.getFinalTransformation();
      transforms[0] = icp_trans * transforms[0];
      std::cout << "ICP pose refinement converged with fitness score: " << reg.getFitnessScore() << std::endl;
    } else {
      std::cout << "ICP pose refinement did not converge." << std::endl;
    }
  }

  //
  // Hypothesis verification
  //
  //  std::vector<bool> hypotheses_mask; // Mask Vector to identify positive hypotheses
  //  if (false) {
  //    pcl::ScopeTime t("Hypothesis verification");
  //
  //    //Algorithm params 
  //    int icp_max_iter_(5);
  //    float icp_corr_distance_(0.005f);
  //    float hv_clutter_reg_(5.0f);
  //    float hv_inlier_th_(0.005f);
  //    float hv_occlusion_th_(0.01f);
  //    float hv_rad_clutter_(0.03f);
  //    float hv_regularizer_(3.0f);
  //    float hv_rad_normals_(0.05);
  //    bool hv_detect_clutter_(true);
  //
  //
  //    pcl::GlobalHypothesesVerification<PointType, PointType> ghv;
  //    //    ghv.setSceneCloud(object_pc->getObjectPointCloud()); // Scene Cloud
  //    ghv.setSceneCloud(cluster.cloud_); // Scene Cloud
  //    //    ghv.addCompleteModels(model_clouds_aligned); //Models to verify
  //    ghv.addModels(model_clouds_aligned, true); //Models to verify
  //
  //    ghv.setInlierThreshold(hv_inlier_th_);
  //    ghv.setOcclusionThreshold(hv_occlusion_th_);
  //    ghv.setRegularizer(hv_regularizer_);
  //    ghv.setRadiusClutter(hv_rad_clutter_);
  //    ghv.setClutterRegularizer(hv_clutter_reg_);
  //    ghv.setDetectClutter(hv_detect_clutter_);
  //    ghv.setRadiusNormals(hv_rad_normals_);
  //
  //    try {
  //      ghv.verify();
  //      ghv.getMask(hypotheses_mask); // i-element TRUE if hvModels[i] verifies hypotheses
  //    } catch (...) {
  //      ptinf("Hypothesis verification catch all. Returning.");
  //      return false;
  //    }
  //
  //    std::vector<ObjectModel<PointType, NormalType, DescriptorType> > matching_models_temp;
  //    std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > transforms_temp;
  //    LOG4CXX_DEBUG(logger, boost::format("Hypotheses to test: %lu.") % matching_models.size());
  //    for (size_t i = 0; i < matching_models.size(); i++) {
  //      if (!hypotheses_mask[i]) {
  //        LOG4CXX_DEBUG(logger, boost::format("Hypothesis %d failed!") % i);
  //        continue;
  //      }
  //      LOG4CXX_DEBUG(logger, boost::format("Hypothesis %d passed!") % i);
  //
  //      matching_models_temp.push_back(matching_models[i]);
  //      transforms_temp.push_back(transforms[i]);
  //    }
  //
  //    matching_models = matching_models_temp;
  //    transforms = transforms_temp;
  //  }

  //
  //  Output results
  //

  std::cout << "Model instances found: " << transforms.size() << std::endl;
  for (size_t i = 0; i < transforms.size(); ++i) {
    //      std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
    //      std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size() << std::endl;

    printMatrix4f(transforms[i], "transforms[i]");
  }

  //
  //  Visualization
  //
  pcl::visualization::PCLVisualizer viewer("Global Classification");
  //viewer.addCoordinateSystem();

  pcl::PointCloud<pcl::PointXYZ>::Ptr model_viewpoint(new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PointXYZ point;
  point.x = matching_model.cloud_->sensor_origin_(0);
  point.y = matching_model.cloud_->sensor_origin_(1);
  point.z = matching_model.cloud_->sensor_origin_(2);
  model_viewpoint->push_back(point);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> model_viewpoint_color_handler(model_viewpoint, 0, 255, 0);
  viewer.addPointCloud(model_viewpoint, model_viewpoint_color_handler, "model_viewpoint");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "model_viewpoint");


  pcl::PointCloud<pcl::PointXYZ>::Ptr scene_viewpoint(new pcl::PointCloud<pcl::PointXYZ> ());
  //  pcl::PointXYZ point;
  point.x = cluster.cloud_->sensor_origin_(0);
  point.y = cluster.cloud_->sensor_origin_(1);
  point.z = cluster.cloud_->sensor_origin_(2);
  scene_viewpoint->push_back(point);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> scene_viewpoint_color_handler(scene_viewpoint, 0, 0, 255);
  viewer.addPointCloud(scene_viewpoint, scene_viewpoint_color_handler, "scene_viewpoint");
  viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "scene_viewpoint");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> model_color_handler(matching_model.cloud_, 0, 255, 0);
  viewer.addPointCloud(matching_model.cloud_, model_color_handler, "model_cloud");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler(cluster.cloud_, 0, 0, 255);
  viewer.addPointCloud(cluster.cloud_, scene_color_handler, "scene_cloud");

  if (show_matches_) {

    for (size_t i = 0; i < transforms.size(); ++i) {
      pcl::PointCloud<PointType>::Ptr pose_aligned_cluster(new pcl::PointCloud<PointType> ());
      //      Eigen::Matrix4f rototranslation_inv = transforms[i].inverse();
      //      pcl::transformPointCloud(*cluster.cloud_, *pose_aligned_cluster, rototranslation_inv);
      pcl::transformPointCloud(*matching_model.cloud_, *pose_aligned_cluster, transforms[i]);

      std::stringstream ss_cloud;
      ss_cloud << "instance" << i;

      pcl::visualization::PointCloudColorHandlerCustom<PointType> pose_aligned_scene_color_handler(pose_aligned_cluster, 255, 0, 0);
      viewer.addPointCloud(pose_aligned_cluster, pose_aligned_scene_color_handler, ss_cloud.str());

      std::stringstream ss_cloud_viewpoint;
      ss_cloud_viewpoint << "viewpoint_instance" << i;

      pcl::PointCloud<pcl::PointXYZ>::Ptr pose_aligned_viewpoint(new pcl::PointCloud<pcl::PointXYZ> ());
      pcl::PointXYZ point;
      point.x = pose_aligned_cluster->sensor_origin_(0);
      point.y = pose_aligned_cluster->sensor_origin_(1);
      point.z = pose_aligned_cluster->sensor_origin_(2);
      pose_aligned_viewpoint->push_back(point);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> pose_aligned_viewpoint_color_handler(model_viewpoint, 255, 0, 0);
      viewer.addPointCloud(pose_aligned_viewpoint, pose_aligned_viewpoint_color_handler, ss_cloud_viewpoint.str());
      viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, ss_cloud_viewpoint.str());
    }
  }

  while (!viewer.wasStopped()) {
    viewer.spinOnce();
  }

  return (0);
}
