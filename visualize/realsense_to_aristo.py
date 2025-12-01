import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import sapien
import cv2
import numpy as np
import pyrealsense2 as rs
from aristo.AristoUtils import AristoUtils
from HandTrackingModule.HandTracking import HandTracking as ht

class HandKinematics:
    def __init__(self):

        scene = sapien.Scene()
        scene.add_ground(0)

        scene.set_ambient_light([0.5, 0.5, 0.5])
        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        self.viewer = scene.create_viewer()
        self.viewer.set_camera_xyz(x=.3, y=.2, z=.5)
        self.viewer.set_camera_rpy(r=0, p=-0.4, y=np.pi*3/4)

        self.scene = scene

        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        self.hand = loader.load("aristo/robot.urdf")
        self.hand.set_root_pose(sapien.Pose([0, 0, .3], [1, 0, 0, 0]))

        self.target_positions = [0.0] * self.hand.dof

    def update_target_positions(self, new_positions):
        self.target_positions = new_positions

    def update(self):
        for _ in range(4):
            for link in self.hand.links:
                link.disable_gravity = True
            self.scene.step()  # advance simulation to update kinematics
        active_joints = self.hand.get_active_joints()
        # print(self.hand.joint_names)
        self.hand.set_qpos(self.target_positions)
        print(self.target_positions)
        self.scene.update_render()
        self.viewer.render()
        

class CamCampture:
    def __init__(self):
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            profile = self.pipeline.start(config)
        except Exception as e:
            print("Failed to start RealSense pipeline:", e)
            return

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # Get camera intrinsics for deprojection
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_profile.get_intrinsics()
        self.aristo = AristoUtils()
        self.detector = ht()

    def update(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Feed color image to detector
        img = self.detector.findHands(color_image.copy())
        data_left, data_right = self.detector.findNormalizedPosition(img)
        aristo_data = self.process_hand_data(data_right)
        return img, aristo_data

    def process_hand_data(self, data_right):
        if data_right.shape == (21, 3):
            # Orient the point cloud for viewing
            data_right = self.detector.anchor_hand(data_right)
            data_right = self.detector.reorient_hand(data_right)
            angles = self.aristo.get_angles(data_right)
            return angles
        return None



if __name__ == "__main__":
    balance_passive_force = True
    hand_kin = HandKinematics()
    cam = CamCampture()
    while not hand_kin.viewer.closed:
        img, aristo_data = cam.update()
        if aristo_data is not None:
            hand_kin.update_target_positions(aristo_data)
        hand_kin.update()
        cv2.imshow("MediaPipe Hands", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break