import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from HandTrackingModule.HandTracking import HandTracking as ht
import pyrealsense2 as rs
import cv2
import numpy as np
import open3d as o3d

# Initialize RealSense pipeline
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
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(mesh_frame)

    # Initialize left and right hand point clouds
    pcd_left = o3d.geometry.PointCloud()
    pcd_left.paint_uniform_color([1, 0.706, 0])
    vis.add_geometry(pcd_left)

    pcd_right = o3d.geometry.PointCloud()
    pcd_right.paint_uniform_color([0, 0.651, 0.929])
    vis.add_geometry(pcd_right)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Feed color image to detector
            img = detector.findHands(color_image.copy())

            # Prepare arrays for left/right hand 3D landmarks
            left_points = []
            right_points = []

            h, w, _ = color_image.shape

            if getattr(detector, 'results', None) and detector.results.multi_hand_landmarks:
                for hand_idx, landmarks in enumerate(detector.results.multi_hand_landmarks):
                    handedness = detector.results.multi_handedness[hand_idx].classification[0].index
                    pts3d = []
                    for lm in landmarks.landmark:
                        px = min(max(int(lm.x * w), 0), w - 1)
                        py = min(max(int(lm.y * h), 0), h - 1)

                        # Get depth in meters
                        z = depth_frame.get_distance(px, py)
                        if z == 0:
                            # Try nearby pixels to mitigate missing depth
                            neighbours = [(0,0),(1,0),(-1,0),(0,1),(0,-1),(2,0),(-2,0),(0,2),(0,-2)]
                            for dx,dy in neighbours:
                                nx = min(max(px+dx,0), w-1)
                                ny = min(max(py+dy,0), h-1)
                                z = depth_frame.get_distance(nx, ny)
                                if z > 0:
                                    px, py = nx, ny
                                    break

                        if z == 0:
                            # skip this landmark if no depth available
                            pts3d.append([0.0,0.0,0.0])
                            continue

                        point = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], z)
                        pts3d.append(point)

                    pts3d = np.array(pts3d)
                    if handedness == 1:
                        left_points = pts3d
                    else:
                        right_points = pts3d

            # Update Open3D point clouds
            if isinstance(right_points, np.ndarray) and right_points.shape == (21,3):
                try:
                    vis.remove_geometry(pcd_right)
                except:
                    pass
                pcd_right = o3d.geometry.PointCloud()
                pcd_right.points = o3d.utility.Vector3dVector(right_points)
                pcd_right.paint_uniform_color([1, 0, 0])
                vis.add_geometry(pcd_right)

            if isinstance(left_points, np.ndarray) and left_points.shape == (21,3):
                try:
                    vis.remove_geometry(pcd_left)
                except:
                    pass
                pcd_left = o3d.geometry.PointCloud()
                pcd_left.points = o3d.utility.Vector3dVector(left_points)
                pcd_left.paint_uniform_color([0, 0, 1])
                vis.add_geometry(pcd_left)

            # Update visualization
            vis.poll_events()
            vis.update_renderer()

            img = detector.displayFPS(img)
            cv2.imshow("RealSense - MediaPipe Hands", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop pipeline and close windows
        pipeline.stop()
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    


