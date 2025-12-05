import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from HandTrackingModule.HandTracking import HandTracking as ht
from HandTrackingModule.Vis3D import Vis3D
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs

def main():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = pipeline.start(config)
    except Exception as e:
        print("Failed to start RealSense pipeline:", e)
        return

    align_to = rs.stream.color
    align = rs.align(align_to)

    # Get camera intrinsics for deprojection
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_profile.get_intrinsics()

    detector = ht()

    # Create Open3D visualization window
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    vis = Vis3D()
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # vis.add_geometry(mesh_frame)

    # # Initialize left and right hand point clouds
    # pcd_left = o3d.geometry.PointCloud()
    # pcd_left.paint_uniform_color([1, 0.706, 0])
    # vis.add_geometry(pcd_left)

    # pcd_right = o3d.geometry.PointCloud()
    # pcd_right.paint_uniform_color([0, 0.651, 0.929])
    # vis.add_geometry(pcd_right)


    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Feed color image to detector
        img = detector.findHands(color_image.copy())
        data_left, data_right = detector.findNormalizedPosition(img)

        # Create Open3D point clouds for the left and right hands if they exist
        if data_right.shape == (21, 3):
            # Orient the point cloud for viewing
            data_right = detector.anchor_hand(data_right)
            data_right = detector.reorient_hand(data_right)
            angle = detector.calculate_angle(data_right[5], data_right[6], data_right[8])
            vis.show_hand(data_right, color=vis.red)
            vis.visualize_angle_projection(
                origin=data_right[0],
                vec_a=data_right[2] - data_right[0],
                vec_b=data_right[5] - data_right[0],
                plane_normal=data_right[9] - data_right[0],
                radius=0.03,
                color=[0.2, 0.8, 0.2],
                vector_color=[1, 0, 0],
                proj_color=[0, 0, 1],
                n_points=32
            )
            # vis.remove_geometry(pcd_right)
            # pcd_right = o3d.geometry.PointCloud()
            # pcd_right.points = o3d.utility.Vector3dVector(data_right)
            # pcd_right.paint_uniform_color([1, 0, 0])
            # vis.add_geometry(pcd_right)

            img = detector.display_angle(img, angle)
        # if data_left.shape == (21, 3):
        #     vis.remove_geometry(pcd_left)
        #     pcd_left = o3d.geometry.PointCloud()
        #     pcd_left.points = o3d.utility.Vector3dVector(data_left)
        #     pcd_left.paint_uniform_color([0, 0, 1])
  
        #     vis.add_geometry(pcd_left)

  
  

        # Update visualization
        vis.poll_events()
        vis.update_renderer()

        img = detector.displayFPS(img)
        cv2.imshow("MediaPipe Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close Open3D visualization window
    vis.destroy_window()


if __name__ == "__main__":
    main()
    


