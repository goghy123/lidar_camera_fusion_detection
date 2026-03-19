#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <yolo_msgs/msg/detection_array.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/crop_box.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <opencv2/opencv.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <thread>
#include <vector>
#include <mutex>





class LidarCameraFusionNode : public rclcpp::Node
{
public:
    LidarCameraFusionNode()
        : Node("filter_fusion_detection_node"),
          tf_buffer_(this->get_clock()),  // 初始化TF2缓冲区（用于坐标变换）
          tf_listener_(tf_buffer_)        // 初始化TF2监听器
    {
        declare_parameters();  // 声明并加载参数
        initialize_subscribers_and_publishers();  // 设置订阅者和发布者
    }

private:
    // 边界框结构体，用于存储检测框信息
    struct BoundingBox {
        double x_min, y_min, x_max, y_max;  // 图像空间中的边界框坐标
        double sum_x = 0, sum_y = 0, sum_z = 0;  // 累积的点坐标（用于求平均值）
        int count = 0;  // 边界框内的点数
        bool valid = false;  // 标志位：边界框是否有效
        int id = -1;  // 检测到的目标ID
        std::string class_name; // 类别名称
        float score = 0.0;      // 置信度
        pcl::PointCloud<pcl::PointXYZ>::Ptr object_cloud = nullptr;  // 目标的点云数据
    };

    // 从参数服务器声明并加载参数
    void declare_parameters()
    {
        declare_parameter<std::string>("lidar_frame", "x500_lidar_camera_1/lidar_link/gpu_lidar");
        declare_parameter<std::string>("camera_frame", "observer/gimbal_camera");
        declare_parameter<float>("min_range", 0.2);
        declare_parameter<float>("max_range", 10.0);

        get_parameter("lidar_frame", lidar_frame_);
        get_parameter("camera_frame", camera_frame_);
        get_parameter("min_range", min_range_);
        get_parameter("max_range", max_range_);

        RCLCPP_INFO(
            get_logger(),
            "参数: lidar_frame='%s', camera_frame='%s', min_range=%.2f, max_range=%.2f",
            lidar_frame_.c_str(),
            camera_frame_.c_str(),
            min_range_,
            max_range_
        );
    }

    // 初始化订阅者和发布者
    void initialize_subscribers_and_publishers()
    {
        // 订阅点云、图像和检测结果
        point_cloud_sub_.subscribe(this, "/scan/points");
        image_sub_.subscribe(this, "/observer/gimbal_camera");
        detection_sub_.subscribe(this, "/rgb/tracking");
        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/observer/gimbal_camera_info", 10, std::bind(&LidarCameraFusionNode::camera_info_callback, this, std::placeholders::_1));

        // 同步器：用于对齐点云、图像和检测消息
        using SyncPolicy = message_filters::sync_policies::ApproximateTime<
            sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>;
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(SyncPolicy(10), point_cloud_sub_, image_sub_, detection_sub_);
        sync_->registerCallback(std::bind(&LidarCameraFusionNode::sync_callback, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));

        // 发布融合图像、目标位姿和目标点云
        image_publisher_ = create_publisher<sensor_msgs::msg::Image>("/image_lidar_fusion", 10);
        pose_publisher_ = create_publisher<geometry_msgs::msg::PoseArray>("/detected_object_pose", 10);
        object_point_cloud_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/detected_object_point_cloud", 10);
        // 3D框发布者
        marker_publisher_ = create_publisher<visualization_msgs::msg::MarkerArray>("/detected_object_markers", 10);
    }

    // 相机信息回调函数：初始化相机模型
    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        camera_model_.fromCameraInfo(msg);  // 加载相机内参
        image_width_ = msg->width;  // 存储图像宽度
        image_height_ = msg->height;  // 存储图像高度
    }

    // 同步回调函数：处理点云、图像和检测结果
    void sync_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& point_cloud_msg,
                       const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
                       const yolo_msgs::msg::DetectionArray::ConstSharedPtr& detection_msg)
    {

        // RCLCPP_INFO(this->get_logger(), "同步回调函数已触发");

        // 处理点云：裁剪并转换到相机坐标系
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_camera_frame = processPointCloud(point_cloud_msg);
        if (!cloud_camera_frame) {
           RCLCPP_ERROR(get_logger(), "点云处理失败。退出回调函数。");
            return;
        }

        // 处理检测结果：提取边界框
        std::vector<BoundingBox> bounding_boxes = processDetections(detection_msg);

        // 将3D点投影到2D图像空间，并与边界框关联
        std::vector<cv::Point2d> projected_points = projectPointsAndAssociateWithBoundingBoxes(cloud_camera_frame, bounding_boxes);

        // 在LiDAR坐标系中计算目标位姿
        geometry_msgs::msg::PoseArray pose_array = calculateObjectPoses(bounding_boxes, point_cloud_msg->header.stamp);

        // 发布结果：融合图像、目标位姿和目标点云
        publishResults(image_msg, projected_points, bounding_boxes, pose_array);

        //发布 3D Markers
        publish3DMarkers(bounding_boxes, point_cloud_msg->header.stamp);
    }

    // 处理点云：裁剪并转换到相机坐标系
    pcl::PointCloud<pcl::PointXYZ>::Ptr processPointCloud(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& point_cloud_msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*point_cloud_msg, *cloud);  // 将ROS消息转换为PCL点云

        // 将点云裁剪到定义的范围
        pcl::CropBox<pcl::PointXYZ> box_filter;
        box_filter.setInputCloud(cloud);
        box_filter.setMin(Eigen::Vector4f(min_range_, -max_range_, -max_range_, 1.0f));
        box_filter.setMax(Eigen::Vector4f(max_range_, max_range_, max_range_, 1.0f));
        box_filter.filter(*cloud);

        // 使用TF2将点云转换到相机坐标系
        rclcpp::Time cloud_time(point_cloud_msg->header.stamp);

        // 将点云转换到相机坐标系
         if (cloud->empty()) {
            RCLCPP_WARN(get_logger(), "过滤后点云为空，跳过坐标变换。");
            return cloud;
        }

        if (tf_buffer_.canTransform(camera_frame_, cloud->header.frame_id, cloud_time, tf2::durationFromSec(1.0))) {
            geometry_msgs::msg::TransformStamped transform = tf_buffer_.lookupTransform(camera_frame_, cloud->header.frame_id, cloud_time, tf2::durationFromSec(1.0));
            Eigen::Affine3d eigen_transform = tf2::transformToEigen(transform); // Eigen::Affine3d - 这是一个4x4变换矩阵
            pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::transformPointCloud(*cloud, *transformed_cloud, eigen_transform);
            return transformed_cloud;
        } else {
            RCLCPP_ERROR(get_logger(), "无法将点云从 %s 转换到 %s", cloud->header.frame_id.c_str(), camera_frame_.c_str());
            return nullptr;
        }
    }

    // 处理检测结果：从YOLO检测中提取边界框
    std::vector<BoundingBox> processDetections(const yolo_msgs::msg::DetectionArray::ConstSharedPtr& detection_msg)
    {
        std::vector<BoundingBox> bounding_boxes;
        for (const auto& detection : detection_msg->detections) {
            BoundingBox bbox;
            bbox.x_min = detection.bbox.center.position.x - detection.bbox.size.x / 2.0;
            bbox.y_min = detection.bbox.center.position.y - detection.bbox.size.y / 2.0;
            bbox.x_max = detection.bbox.center.position.x + detection.bbox.size.x / 2.0;
            bbox.y_max = detection.bbox.center.position.y + detection.bbox.size.y / 2.0;
            bbox.valid = true;
            try {
                bbox.id = std::stoi(detection.id);  // 将检测ID转换为整数
            } catch (const std::exception& e) {
                RCLCPP_ERROR(get_logger(), "无法将检测ID转换为整数: %s", e.what());
                continue;
            }
            // 给标签和置信度赋值
            try {
                bbox.class_name = detection.class_name; 
                bbox.score = detection.score; 
            } catch (const std::exception& e) {
                RCLCPP_ERROR(get_logger(), "无法将标签和置信度赋值");
                continue;
            }
            

            bbox.object_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
            bounding_boxes.push_back(bbox);
        }
        return bounding_boxes;
    }

    // 将3D点投影到2D图像空间，并与边界框关联
    std::vector<cv::Point2d> projectPointsAndAssociateWithBoundingBoxes(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_camera_frame,
        std::vector<BoundingBox>& bounding_boxes)
    {
        std::vector<cv::Point2d> projected_points;
        if (!cloud_camera_frame){
            RCLCPP_WARN(get_logger(), "在projectPointsAndAssociateWithBoundingBoxes中点云无效。跳过投影。");
            return projected_points;
        }

        // Lambda函数：并行处理点
        auto process_points = [&](size_t start, size_t end) {
            for (size_t i = start; i < end; ++i) {
                const auto& point = cloud_camera_frame->points[i];
                if (point.z > 0) {
                    cv::Point3d pt_cv(point.x, point.y, point.z);
                    cv::Point2d uv = camera_model_.project3dToPixel(pt_cv);

                    for (auto& bbox : bounding_boxes) {
                        if (uv.x >= bbox.x_min && uv.x <= bbox.x_max &&
                            uv.y >= bbox.y_min && uv.y <= bbox.y_max) {
                            // 点位于边界框内
                            std::lock_guard<std::mutex> lock(mtx);  // 确保线程安全更新
                            projected_points.push_back(uv);  // 将投影点添加到结果中
                            bbox.sum_x += point.x;  // 累加点坐标（以米为单位）
                            bbox.sum_y += point.y;
                            bbox.sum_z += point.z;
                            bbox.count++;  // 增加点计数
                            bbox.object_cloud->points.push_back(point);  // 将点添加到目标点云
                            break;  // 提前退出：跳过此点的其余边界框
                        }
                    }
                }
            }
        };

        // 将工作分配到多个线程
        const size_t num_threads = std::thread::hardware_concurrency();
        const size_t points_per_thread = cloud_camera_frame->points.size() / num_threads;
        std::vector<std::thread> threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t start = t * points_per_thread;
            size_t end = (t == num_threads - 1) ? cloud_camera_frame->points.size() : start + points_per_thread;
            threads.emplace_back(process_points, start, end);
        }

        // 等待所有线程完成
        for (auto& thread : threads) {
            thread.join();
        }

        return projected_points;
    }

    // 在LiDAR坐标系中计算目标位姿
    geometry_msgs::msg::PoseArray calculateObjectPoses(
        const std::vector<BoundingBox>& bounding_boxes,
        const rclcpp::Time& cloud_time)
    {
        geometry_msgs::msg::PoseArray pose_array;
        pose_array.header.stamp = cloud_time;
        pose_array.header.frame_id = lidar_frame_;

        // 查找从相机到LiDAR坐标系的变换
        geometry_msgs::msg::TransformStamped transform;
        try {
            transform = tf_buffer_.lookupTransform(lidar_frame_, camera_frame_, cloud_time, tf2::durationFromSec(1.0));
        } catch (tf2::TransformException& ex) {
            RCLCPP_ERROR(get_logger(), "查找变换失败: %s", ex.what());
            return pose_array;  // 如果变换失败，返回空的PoseArray
        }

        // 将变换转换为Eigen格式以加快计算
        Eigen::Affine3d eigen_transform = tf2::transformToEigen(transform);

        // 计算每个边界框的平均位置，并转换到LiDAR坐标系
        for (const auto& bbox : bounding_boxes) {
            if (bbox.count > 0) {
                double avg_x = bbox.sum_x / bbox.count;
                double avg_y = bbox.sum_y / bbox.count;
                double avg_z = bbox.sum_z / bbox.count;

                // 在相机坐标系中创建位姿
                geometry_msgs::msg::PoseStamped pose_camera;
                pose_camera.header.stamp = cloud_time;
                pose_camera.header.frame_id = camera_frame_;
                pose_camera.pose.position.x = avg_x;
                pose_camera.pose.position.y = avg_y;
                pose_camera.pose.position.z = avg_z;
                pose_camera.pose.orientation.w = 1.0;

                // 将位姿转换到LiDAR坐标系
                try {
                    geometry_msgs::msg::PoseStamped pose_lidar = tf_buffer_.transform(pose_camera, lidar_frame_, tf2::durationFromSec(1.0));
                    pose_array.poses.push_back(pose_lidar.pose);
                } catch (tf2::TransformException& ex) {
                    RCLCPP_ERROR(get_logger(), "位姿变换失败: %s", ex.what());
                }
            } else {
                 RCLCPP_WARN(get_logger(), "跳过边界框ID %d的位姿计算，计数为0", bbox.id);
            }
        }
        return pose_array;
    }

    // 生成并发布 3D 检测框和标签 Marker
    void publish3DMarkers(
        const std::vector<BoundingBox>& bounding_boxes,
        const rclcpp::Time& cloud_time)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        
        int marker_id = 0;

        // 2. 遍历每个 bbox
        for (const auto& bbox : bounding_boxes) {
            // 只处理有点云数据的 bbox
            if (bbox.count > 0 && !bbox.object_cloud->empty()) {
                
                // 3. 将计算相机坐标系下的 Min/Max
                pcl::PointXYZ min_pt, max_pt;
                pcl::getMinMax3D(*bbox.object_cloud, min_pt, max_pt);
                

                // 计算中心点和尺寸
                double center_x = (min_pt.x + max_pt.x) / 2.0;
                double center_y = (min_pt.y + max_pt.y) / 2.0;
                double center_z = (min_pt.z + max_pt.z) / 2.0;
                double dim_x = max_pt.x - min_pt.x;
                double dim_y = max_pt.y - min_pt.y;
                double dim_z = max_pt.z - min_pt.z;

                // 防止尺寸为0导致Rviz报错
                if(dim_x < 0.01) dim_x = 0.01;
                if(dim_y < 0.01) dim_y = 0.01;
                if(dim_z < 0.01) dim_z = 0.01;

                // --- 创建 3D 框 Marker (CUBE) ---
                visualization_msgs::msg::Marker box_marker;
                box_marker.header.frame_id = camera_frame_;
                box_marker.header.stamp = cloud_time;
                box_marker.ns = "detection_boxes";
                box_marker.id = marker_id++;
                box_marker.type = visualization_msgs::msg::Marker::CUBE;
                box_marker.action = visualization_msgs::msg::Marker::ADD;
                
                box_marker.pose.position.x = center_x;
                box_marker.pose.position.y = center_y;
                box_marker.pose.position.z = center_z;
                box_marker.pose.orientation.w = 1.0; // 无旋转，轴对齐的包围盒 (AABB)

                box_marker.scale.x = dim_x;
                box_marker.scale.y = dim_y;
                box_marker.scale.z = dim_z;

                box_marker.color.r = 0.0f;
                box_marker.color.g = 1.0f;
                box_marker.color.b = 0.0f;
                box_marker.color.a = 0.3f; // 半透明绿色
                box_marker.lifetime = rclcpp::Duration::from_seconds(0.2); // 短生命周期，自动消失

                marker_array.markers.push_back(box_marker);

                // --- 创建 文本 Marker (TEXT) ---
                visualization_msgs::msg::Marker text_marker;
                text_marker.header.frame_id = camera_frame_;
                text_marker.header.stamp = cloud_time;
                text_marker.ns = "detection_info";
                text_marker.id = marker_id++;
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.action = visualization_msgs::msg::Marker::ADD;

                // 放在框的上方
                text_marker.pose.position.x = center_x;
                text_marker.pose.position.y = center_y;
                text_marker.pose.position.z = max_pt.z + 0.3; // 框顶上方 0.3米
                text_marker.pose.orientation.w = 1.0;

                text_marker.scale.z = 0.3; // 文字高度

                text_marker.color.r = 1.0f;
                text_marker.color.g = 1.0f;
                text_marker.color.b = 1.0f;
                text_marker.color.a = 1.0f;
                text_marker.lifetime = rclcpp::Duration::from_seconds(0.2);

                // 格式化字符串: "Car: 0.95"
                std::stringstream ss;
                ss << bbox.class_name << ": " << std::fixed << std::setprecision(2) << bbox.score;
                text_marker.text = ss.str();

                marker_array.markers.push_back(text_marker);
            }
        }


        // 发布 MarkerArray
        marker_publisher_->publish(marker_array);
    }


    // 发布结果：融合图像、目标位姿和目标点云
    void publishResults(
        const sensor_msgs::msg::Image::ConstSharedPtr& image_msg,
        const std::vector<cv::Point2d>& projected_points,
        const std::vector<BoundingBox>& bounding_boxes,
        const geometry_msgs::msg::PoseArray& pose_array)
    {
        // 在图像上绘制投影点
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
        for (const auto& uv : projected_points) {
            cv::circle(cv_ptr->image, cv::Point(uv.x, uv.y), 5, CV_RGB(255, 0, 0), -1);
        }

        // 发布融合图像
        image_publisher_->publish(*cv_ptr->toImageMsg());

        // 将所有目标点云合并为一个点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr combined_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& bbox : bounding_boxes) {
            if (bbox.count > 0 && bbox.object_cloud) {
                *combined_cloud += *bbox.object_cloud;  // 拼接点云
            }
        }

        // 发布合并的点云
        if (!combined_cloud->empty()) {
            sensor_msgs::msg::PointCloud2 combined_cloud_msg;
            pcl::toROSMsg(*combined_cloud, combined_cloud_msg);
            combined_cloud_msg.header = image_msg->header;
            combined_cloud_msg.header.frame_id = camera_frame_;
            object_point_cloud_publisher_->publish(combined_cloud_msg);
        }

        // 发布目标位姿
        pose_publisher_->publish(pose_array);
    }

    // TF2缓冲区和监听器（用于坐标变换）
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // 相机模型（用于将3D点投影到2D图像空间）
    image_geometry::PinholeCameraModel camera_model_;

    // 裁剪和坐标系参数
    float min_range_, max_range_;
    std::string camera_frame_, lidar_frame_;
    int image_width_, image_height_;

    // 点云、图像和检测结果的订阅者
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> point_cloud_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    message_filters::Subscriber<yolo_msgs::msg::DetectionArray> detection_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    // 消息对齐同步器
    std::shared_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::Image, yolo_msgs::msg::DetectionArray>>> sync_;

    // 融合图像、目标位姿、目标点云和YOLO3D框的发布者
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pose_publisher_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_point_cloud_publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher_;

    // 线程安全更新的互斥锁
    std::mutex mtx;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);  // 初始化ROS2
    auto node = std::make_shared<LidarCameraFusionNode>();  // 创建节点
    rclcpp::spin(node);  // 运行节点
    rclcpp::shutdown();  // 关闭ROS2
    return 0;
}
