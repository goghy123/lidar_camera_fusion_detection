#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <yolo_msgs/msg/detection_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sstream>
#include <iomanip>
#include <vector>
#include <array>
#include <limits>


class LidarCameraFusionNode : public rclcpp::Node
{
public:
    LidarCameraFusionNode()
        : Node("filter_fusion_detection_node"),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        declareAndLoadParameters();
        initRegions();
        initSubsAndPubs();
    }

private:
    // =========================================================================
    // 数据结构
    // =========================================================================

    // YOLO 2D 检测框（图像坐标系）
    struct Detection2D {
        double x_min, y_min, x_max, y_max;
        int         id = -1;
        std::string class_name;
        float       score = 0.0f;
    };

    // 点云聚类（相机坐标系）
    struct ClusterInfo {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
        pcl::PointXYZ    min_pt, max_pt;
        Eigen::Vector3d  centroid;
        // 融合结果
        bool        classified   = false;
        std::string class_name;
        float       confidence   = 0.0f;
        int         detection_id = -1;
    };

    // =========================================================================
    // 初始化
    // =========================================================================

    void declareAndLoadParameters()
    {
        declare_parameter<std::string>("lidar_frame",    "x500_lidar_camera_1/lidar_link/gpu_lidar");
        declare_parameter<std::string>("camera_frame",   "observer/gimbal_camera");
        declare_parameter<float>("min_range",            0.2f);
        declare_parameter<float>("max_range",            10.0f);
        declare_parameter<float>("z_axis_min",           -0.5f);
        declare_parameter<float>("z_axis_max",            1.5f);
        declare_parameter<int>("cluster_size_min",       10);
        declare_parameter<int>("cluster_size_max",       2000);
        declare_parameter<int>("region_max",             14);
        declare_parameter<std::string>("sensor_model",   "HDL-32E");
        declare_parameter<float>("iou_threshold",        0.3f);

        get_parameter("lidar_frame",      lidar_frame_);
        get_parameter("camera_frame",     camera_frame_);
        get_parameter("min_range",        min_range_);
        get_parameter("max_range",        max_range_);
        get_parameter("z_axis_min",       z_axis_min_);
        get_parameter("z_axis_max",       z_axis_max_);
        get_parameter("cluster_size_min", cluster_size_min_);
        get_parameter("cluster_size_max", cluster_size_max_);
        get_parameter("region_max",       region_max_);
        get_parameter("sensor_model",     sensor_model_);
        get_parameter("iou_threshold",    iou_threshold_);
    }

    // 按传感器型号初始化径向分区宽度（来自 Python: init_regions）
    void initRegions()
    {
        regions_.assign(static_cast<size_t>(region_max_), 0.0f);

        std::vector<float> r;
        if      (sensor_model_ == "VLP-16")  r = {2,3,3,3,3,3,3,2,3,3,3,3,3,3};
        else if (sensor_model_ == "HDL-32E") r = {4,5,4,5,4,5,5,4,5,4,5,5,4,5};
        else if (sensor_model_ == "HDL-64E") r = {14,14,14,15,14};
        else {
            RCLCPP_FATAL(get_logger(), "未知传感器型号: %s，将使用均匀分区。", sensor_model_.c_str());
            float w = max_range_ / static_cast<float>(region_max_);
            for (int i = 0; i < region_max_; ++i) regions_[i] = w;
            return;
        }
        for (int i = 0; i < std::min(region_max_, (int)r.size()); ++i)
            regions_[i] = r[i];
    }

    void initSubsAndPubs()
    {
        point_cloud_sub_.subscribe(this, "/scan/points");
        image_sub_.subscribe(this, "/observer/gimbal_camera");
        detection_sub_.subscribe(this, "/rgb/tracking");

        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/observer/gimbal_camera_info", 10,
            std::bind(&LidarCameraFusionNode::onCameraInfo, this, std::placeholders::_1));

        using SyncPolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::PointCloud2,
            sensor_msgs::msg::Image,
            yolo_msgs::msg::DetectionArray>;

        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), point_cloud_sub_, image_sub_, detection_sub_);
        sync_->registerCallback(std::bind(&LidarCameraFusionNode::syncCallback,
            this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

        image_publisher_              = create_publisher<sensor_msgs::msg::Image>("/image_lidar_fusion", 10);
        pose_publisher_               = create_publisher<geometry_msgs::msg::PoseArray>("/detected_object_pose", 10);
        object_point_cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/detected_object_point_cloud", 10);
        marker_publisher_             = create_publisher<visualization_msgs::msg::MarkerArray>("/detected_object_markers", 10);
    }

    void onCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        camera_model_.fromCameraInfo(msg);
        image_width_  = static_cast<int>(msg->width);
        image_height_ = static_cast<int>(msg->height);
    }

    // =========================================================================
    // 同步回调主流程
    // =========================================================================

    void syncCallback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr&  cloud_msg,
        const sensor_msgs::msg::Image::ConstSharedPtr&         image_msg,
        const yolo_msgs::msg::DetectionArray::ConstSharedPtr&  det_msg)
    {
        // 1. 点云预处理：裁剪 + 高度过滤 + TF2 转换到相机坐标系
        auto cloud_cam = preprocessPointCloud(cloud_msg);
        if (!cloud_cam || cloud_cam->empty()) return;

        // 2. 分区欧式聚类（在相机坐标系中）
        auto clusters = clusterPointCloud(cloud_cam);

        // 3. 解析 YOLO 2D 检测框
        auto detections = parseDetections(det_msg);

        // 4. 将 3D 聚类框投影到图像平面，IoU 匹配 2D 检测框
        fuseClusterWithDetections(clusters, detections);

        // 5. 发布所有结果
        rclcpp::Time stamp(cloud_msg->header.stamp);
        publishFusedImage(image_msg, clusters);
        publishPoseArray(clusters, stamp);
        publishObjectPointCloud(clusters, cloud_msg->header);
        publish3DMarkers(clusters, stamp);
    }

    // =========================================================================
    // 点云预处理（参考原 C++：裁剪 → 转换到相机坐标系；新增高度过滤）
    // =========================================================================

    pcl::PointCloud<pcl::PointXYZ>::Ptr preprocessPointCloud(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
    {
        auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::fromROSMsg(*msg, *cloud);

        // XY 范围裁剪（同原 C++）
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(cloud);
        crop.setMin(Eigen::Vector4f(min_range_, -max_range_, -max_range_, 1.0f));
        crop.setMax(Eigen::Vector4f(max_range_,  max_range_,  max_range_, 1.0f));
        crop.filter(*cloud);

        // Z 轴高度过滤（来自 Python: z_axis_min / z_axis_max）
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(z_axis_min_, z_axis_max_);
        pass.filter(*cloud);

        if (cloud->empty()) return cloud;

        // TF2 坐标变换：LiDAR → Camera（同原 C++）
        rclcpp::Time stamp(msg->header.stamp);
        if (!tf_buffer_.canTransform(camera_frame_, cloud->header.frame_id,
                                     stamp, tf2::durationFromSec(1.0))) {
            RCLCPP_ERROR(get_logger(), "无法将点云从 %s 转换到 %s",
                         cloud->header.frame_id.c_str(), camera_frame_.c_str());
            return nullptr;
        }
        auto tf = tf_buffer_.lookupTransform(camera_frame_, cloud->header.frame_id,
                                              stamp, tf2::durationFromSec(1.0));
        auto transformed = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        pcl::transformPointCloud(*cloud, *transformed, tf2::transformToEigen(tf));
        return transformed;
    }

    // =========================================================================
    // 分区欧式聚类（来自 Python: divide_into_regions + euclidean_clustering）
    // =========================================================================

    // 将点云按点到原点的径向距离分配到各环形区域
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> divideIntoRegions(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> regions(
            static_cast<size_t>(region_max_));
        for (auto& r : regions)
            r = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

        for (const auto& pt : cloud->points) {
            float dist        = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
            float range_start = 0.0f;
            for (int j = 0; j < region_max_; ++j) {
                float range_end = range_start + regions_[j];
                if (dist > range_start && dist <= range_end) {
                    regions[j]->points.push_back(pt);
                    break;
                }
                range_start = range_end;
            }
        }
        for (auto& r : regions) { r->width = (uint32_t)r->points.size(); r->height = 1; }
        return regions;
    }

    // 对各分区执行欧式聚类，容差随区域编号递增 0.1m
    std::vector<ClusterInfo> clusterPointCloud(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {
        auto regions = divideIntoRegions(cloud);
        std::vector<ClusterInfo> all_clusters;
        float tolerance = 0.0f;

        for (int i = 0; i < region_max_; ++i) {
            tolerance += 0.1f;
            if ((int)regions[i]->points.size() < cluster_size_min_) continue;

            auto tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
            tree->setInputCloud(regions[i]);

            std::vector<pcl::PointIndices> cluster_indices;
            pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
            ec.setClusterTolerance(tolerance);
            ec.setMinClusterSize(cluster_size_min_);
            ec.setMaxClusterSize(cluster_size_max_);
            ec.setSearchMethod(tree);
            ec.setInputCloud(regions[i]);
            ec.extract(cluster_indices);

            for (const auto& indices : cluster_indices) {
                ClusterInfo info;
                info.cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
                for (int idx : indices.indices)
                    info.cloud->points.push_back(regions[i]->points[idx]);
                info.cloud->width  = (uint32_t)info.cloud->points.size();
                info.cloud->height = 1;

                pcl::getMinMax3D(*info.cloud, info.min_pt, info.max_pt);
                Eigen::Vector4f c4;
                pcl::compute3DCentroid(*info.cloud, c4);
                info.centroid = Eigen::Vector3d(c4[0], c4[1], c4[2]);

                all_clusters.push_back(std::move(info));
            }
        }
        return all_clusters;
    }

    // =========================================================================
    // 解析 YOLO 检测结果（同原 C++）
    // =========================================================================

    std::vector<Detection2D> parseDetections(
        const yolo_msgs::msg::DetectionArray::ConstSharedPtr& msg)
    {
        std::vector<Detection2D> dets;
        for (const auto& d : msg->detections) {
            Detection2D det;
            det.x_min      = d.bbox.center.position.x - d.bbox.size.x / 2.0;
            det.y_min      = d.bbox.center.position.y - d.bbox.size.y / 2.0;
            det.x_max      = d.bbox.center.position.x + d.bbox.size.x / 2.0;
            det.y_max      = d.bbox.center.position.y + d.bbox.size.y / 2.0;
            det.class_name = d.class_name;
            det.score      = d.score;
            try { det.id = std::stoi(d.id); } catch (...) { det.id = -1; }
            dets.push_back(det);
        }
        return dets;
    }

    // =========================================================================
    // IoU 融合（来自 Python: fuse_lidar_and_image）
    // 点云已在相机坐标系中，直接用相机模型投影 AABB 角点，与 YOLO 框计算 IoU
    // =========================================================================

    void fuseClusterWithDetections(
        std::vector<ClusterInfo>&       clusters,
        const std::vector<Detection2D>& detections)
    {
        // 8 个角点的 min/max 轴组合索引
        static const std::array<std::array<int,3>, 8> signs = {{
            {0,0,0},{1,0,0},{1,1,0},{0,1,0},
            {0,0,1},{1,0,1},{1,1,1},{0,1,1}
        }};

        for (auto& cluster : clusters) {
            float xs[2] = {cluster.min_pt.x, cluster.max_pt.x};
            float ys[2] = {cluster.min_pt.y, cluster.max_pt.y};
            float zs[2] = {cluster.min_pt.z, cluster.max_pt.z};

            // 投影 8 个角点，计算 2D 投影包围框
            bool   valid = true;
            double px_min = std::numeric_limits<double>::max();
            double px_max = std::numeric_limits<double>::lowest();
            double py_min = std::numeric_limits<double>::max();
            double py_max = std::numeric_limits<double>::lowest();

            for (const auto& s : signs) {
                cv::Point3d pt(xs[s[0]], ys[s[1]], zs[s[2]]);
                if (pt.z <= 0.0) { valid = false; break; }
                cv::Point2d uv = camera_model_.project3dToPixel(pt);
                px_min = std::min(px_min, uv.x); px_max = std::max(px_max, uv.x);
                py_min = std::min(py_min, uv.y); py_max = std::max(py_max, uv.y);
            }
            if (!valid) continue;
            if (px_max < 0 || px_min >= image_width_ || py_max < 0 || py_min >= image_height_) continue;

            // 裁剪到图像边界
            px_min = std::max(px_min, 0.0); px_max = std::min(px_max, (double)(image_width_  - 1));
            py_min = std::max(py_min, 0.0); py_max = std::min(py_max, (double)(image_height_ - 1));

            // 计算 IoU，选最优匹配
            double best_iou = 0.0;
            int    best_idx = -1;

            for (int d = 0; d < (int)detections.size(); ++d) {
                const auto& det = detections[d];
                double ix_l = std::max(px_min, det.x_min);
                double iy_t = std::max(py_min, det.y_min);
                double ix_r = std::min(px_max, det.x_max);
                double iy_b = std::min(py_max, det.y_max);
                if (ix_r <= ix_l || iy_b <= iy_t) continue;

                double inter = (ix_r - ix_l) * (iy_b - iy_t);
                double area1 = (px_max - px_min) * (py_max - py_min);
                double area2 = (det.x_max - det.x_min) * (det.y_max - det.y_min);
                double iou   = inter / (area1 + area2 - inter + 1e-6);

                if (iou > best_iou && iou > iou_threshold_) {
                    best_iou = iou; best_idx = d;
                }
            }

            if (best_idx >= 0) {
                cluster.classified   = true;
                cluster.class_name   = detections[best_idx].class_name;
                cluster.confidence   = detections[best_idx].score;
                cluster.detection_id = detections[best_idx].id;
            }
        }
    }

    // =========================================================================
    // 发布：融合图像（绘制 3D 线框 + 分类标签）
    // =========================================================================

    void publishFusedImage(
        const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
        const std::vector<ClusterInfo>&                clusters)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try { cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8); }
        catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge 转换失败: %s", e.what()); return;
        }
        cv::Mat& img = cv_ptr->image;

        static const std::array<std::array<int,3>, 8> signs = {{
            {0,0,0},{1,0,0},{1,1,0},{0,1,0},
            {0,0,1},{1,0,1},{1,1,1},{0,1,1}
        }};

        for (const auto& cluster : clusters) {
            float xs[2] = {cluster.min_pt.x, cluster.max_pt.x};
            float ys[2] = {cluster.min_pt.y, cluster.max_pt.y};
            float zs[2] = {cluster.min_pt.z, cluster.max_pt.z};

            bool valid = true;
            std::array<cv::Point, 8> pts;
            for (int i = 0; i < 8; ++i) {
                const auto& s = signs[i];
                cv::Point3d pt(xs[s[0]], ys[s[1]], zs[s[2]]);
                if (pt.z <= 0.0) { valid = false; break; }
                cv::Point2d uv = camera_model_.project3dToPixel(pt);
                pts[i] = cv::Point((int)uv.x, (int)uv.y);
            }
            if (!valid) continue;

            cv::Scalar color = cluster.classified ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);

            // 绘制底面、顶面、侧面（来自 Python: cv2.line）
            for (int i = 0; i < 4; ++i) cv::line(img, pts[i],   pts[(i+1)%4],   color, 2);
            for (int i = 0; i < 4; ++i) cv::line(img, pts[i+4], pts[(i+1)%4+4], color, 2);
            for (int i = 0; i < 4; ++i) cv::line(img, pts[i],   pts[i+4],       color, 2);

            if (cluster.classified) {
                cv::Point2d uv = camera_model_.project3dToPixel(
                    cv::Point3d(cluster.centroid.x(), cluster.centroid.y(), cluster.centroid.z()));
                std::ostringstream oss;
                oss << cluster.class_name << ": " << std::fixed << std::setprecision(2) << cluster.confidence;
                cv::putText(img, oss.str(), cv::Point((int)uv.x, (int)uv.y),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
            }
        }
        image_publisher_->publish(*cv_ptr->toImageMsg());
    }

    // =========================================================================
    // 发布：已分类目标的位姿（相机系质心 → TF2 变换到 LiDAR 系，同原 C++）
    // =========================================================================

    void publishPoseArray(
        const std::vector<ClusterInfo>& clusters,
        const rclcpp::Time&             stamp)
    {
        geometry_msgs::msg::PoseArray pose_array;
        pose_array.header.stamp    = stamp;
        pose_array.header.frame_id = lidar_frame_;

        for (const auto& cluster : clusters) {
            if (!cluster.classified) continue;

            geometry_msgs::msg::PoseStamped pose_cam;
            pose_cam.header.stamp    = stamp;
            pose_cam.header.frame_id = camera_frame_;
            pose_cam.pose.position.x    = cluster.centroid.x();
            pose_cam.pose.position.y    = cluster.centroid.y();
            pose_cam.pose.position.z    = cluster.centroid.z();
            pose_cam.pose.orientation.w = 1.0;

            try {
                auto pose_lidar = tf_buffer_.transform(pose_cam, lidar_frame_, tf2::durationFromSec(1.0));
                pose_array.poses.push_back(pose_lidar.pose);
            } catch (tf2::TransformException& ex) {
                RCLCPP_ERROR(get_logger(), "位姿变换失败: %s", ex.what());
            }
        }
        pose_publisher_->publish(pose_array);
    }

    // =========================================================================
    // 发布：已分类目标的合并点云（同原 C++）
    // =========================================================================

    void publishObjectPointCloud(
        const std::vector<ClusterInfo>& clusters,
        const std_msgs::msg::Header&    header)
    {
        auto combined = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        for (const auto& c : clusters)
            if (c.classified && c.cloud) *combined += *c.cloud;
        if (combined->empty()) return;

        sensor_msgs::msg::PointCloud2 out_msg;
        pcl::toROSMsg(*combined, out_msg);
        out_msg.header          = header;
        out_msg.header.frame_id = camera_frame_;
        object_point_cloud_publisher_->publish(out_msg);
    }

    // =========================================================================
    // 发布：3D Marker（线框 + 文字，来自 Python: publish_cluster_markers）
    // =========================================================================

    void publish3DMarkers(
        const std::vector<ClusterInfo>& clusters,
        const rclcpp::Time&             stamp)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        int marker_id = 0;

        auto make_pt = [](float x, float y, float z) {
            geometry_msgs::msg::Point p; p.x=x; p.y=y; p.z=z; return p; };

        for (const auto& cluster : clusters) {
            const auto& mn = cluster.min_pt;
            const auto& mx = cluster.max_pt;

            // 线框 Marker（LINE_LIST，同 Python: Marker.LINE_LIST）
            visualization_msgs::msg::Marker wire;
            wire.header.frame_id    = camera_frame_;
            wire.header.stamp       = stamp;
            wire.ns                 = "detection_boxes";
            wire.id                 = marker_id++;
            wire.type               = visualization_msgs::msg::Marker::LINE_LIST;
            wire.action             = visualization_msgs::msg::Marker::ADD;
            wire.scale.x            = 0.05;
            wire.lifetime           = rclcpp::Duration::from_seconds(0.2);

            // 颜色策略（同 Python: publish_cluster_markers 的颜色逻辑）
            if (!cluster.classified) {
                wire.color = {0.0f, 1.0f, 0.5f, 0.8f};
            } else {
                const auto& cn = cluster.class_name;
                if      (cn=="person"  || cn=="pedestrian")                       wire.color = {1.0f,0.0f,0.0f,1.0f};
                else if (cn=="car"     || cn=="vehicle" || cn=="truck"||cn=="bus") wire.color = {0.0f,0.0f,1.0f,1.0f};
                else if (cn=="bicycle" || cn=="bike")                             wire.color = {1.0f,0.5f,0.0f,1.0f};
                else                                                               wire.color = {0.8f,0.2f,0.8f,1.0f};
            }

            // 12 条边，24 个端点（同 Python: p[0..23]）
            std::array<geometry_msgs::msg::Point, 24> p;
            p[0] =make_pt(mx.x,mx.y,mx.z); p[1] =make_pt(mn.x,mx.y,mx.z);
            p[2] =make_pt(mx.x,mx.y,mx.z); p[3] =make_pt(mx.x,mn.y,mx.z);
            p[4] =make_pt(mx.x,mx.y,mx.z); p[5] =make_pt(mx.x,mx.y,mn.z);
            p[6] =make_pt(mn.x,mn.y,mn.z); p[7] =make_pt(mx.x,mn.y,mn.z);
            p[8] =make_pt(mn.x,mn.y,mn.z); p[9] =make_pt(mn.x,mx.y,mn.z);
            p[10]=make_pt(mn.x,mn.y,mn.z); p[11]=make_pt(mn.x,mn.y,mx.z);
            p[12]=make_pt(mn.x,mx.y,mx.z); p[13]=make_pt(mn.x,mx.y,mn.z);
            p[14]=make_pt(mn.x,mx.y,mx.z); p[15]=make_pt(mn.x,mn.y,mx.z);
            p[16]=make_pt(mx.x,mn.y,mx.z); p[17]=make_pt(mx.x,mn.y,mn.z);
            p[18]=make_pt(mx.x,mn.y,mx.z); p[19]=make_pt(mn.x,mn.y,mx.z);
            p[20]=make_pt(mx.x,mx.y,mn.z); p[21]=make_pt(mn.x,mx.y,mn.z);
            p[22]=make_pt(mx.x,mx.y,mn.z); p[23]=make_pt(mx.x,mn.y,mn.z);
            for (const auto& pt : p) wire.points.push_back(pt);
            marker_array.markers.push_back(wire);

            // 文字 Marker（仅已分类目标）
            if (cluster.classified) {
                visualization_msgs::msg::Marker text;
                text.header.frame_id    = camera_frame_;
                text.header.stamp       = stamp;
                text.ns                 = "detection_info";
                text.id                 = marker_id++;
                text.type               = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text.action             = visualization_msgs::msg::Marker::ADD;
                text.pose.position.x    = (mn.x + mx.x) / 2.0;
                text.pose.position.y    = (mn.y + mx.y) / 2.0;
                text.pose.position.z    = mx.z + 0.3;
                text.pose.orientation.w = 1.0;
                text.scale.z            = 0.3;
                text.color              = {1.0f, 1.0f, 1.0f, 1.0f};
                text.lifetime           = rclcpp::Duration::from_seconds(0.2);

                std::ostringstream oss;
                oss << cluster.class_name << ": " << std::fixed << std::setprecision(2) << cluster.confidence;
                text.text = oss.str();
                marker_array.markers.push_back(text);
            }
        }
        marker_publisher_->publish(marker_array);
    }

    // =========================================================================
    // 成员变量
    // =========================================================================

    tf2_ros::Buffer            tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    image_geometry::PinholeCameraModel camera_model_;
    int image_width_  = 640;
    int image_height_ = 480;

    std::string lidar_frame_, camera_frame_, sensor_model_;
    float min_range_, max_range_;
    float z_axis_min_, z_axis_max_;
    int   cluster_size_min_, cluster_size_max_;
    int   region_max_;
    float iou_threshold_;
    std::vector<float> regions_;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2>    point_cloud_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image>           image_sub_;
    message_filters::Subscriber<yolo_msgs::msg::DetectionArray>    detection_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr   camera_info_sub_;

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr             image_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr        pose_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr        object_point_cloud_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  marker_publisher_;
};


int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarCameraFusionNode>());
    rclcpp::shutdown();
    return 0;
}
