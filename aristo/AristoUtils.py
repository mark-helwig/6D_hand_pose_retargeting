import numpy as np



class AristoUtils:
    def __calculate_angle(self, a, b, c):
        """Calculate the angle ABC (in degrees) given three points a, b, and c."""
        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return angle
    
    def __calculate_angle_from_vectors(self, v1a, v1b, v2a, v2b):
        """Calculate the angle between two vectors v1 and v2 defined by points (v1a, v1b) and (v2a, v2b)."""
        vec1 = v1b - v1a
        vec2 = v2b - v2a

        cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return angle
    
    def __define_plane(self, p1, p2, p3):
        # Create two vectors in the plane
        v1 = p2 - p1
        v2 = p3 - p1

        # Compute the normal vector using the cross product
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

        # The plane can be defined by the normal vector and a point (p1)
        d = -np.dot(normal, p1)

        return normal, d
    
    def __project_angle_onto_plane(self, angle, plane_normal):
        angle_vector = np.array([np.cos(angle), np.sin(angle), 0])  # Assuming angles are in radians
        projection = angle_vector - np.dot(angle_vector, plane_normal) * plane_normal
        projected_angle = np.arctan2(projection[1], projection[0])  # Convert back to angle
        return projected_angle
    
    def get_index_angles(self, hand_data):

        angles = {}
        # Calculate angles for the index finger joints
        angles['distal'] = -1.0 * (self.__calculate_angle(hand_data[5], hand_data[6], hand_data[8]) - np.pi)
        angles['proximal'] = -1.0 * (self.__calculate_angle(hand_data[0], hand_data[5], hand_data[6]) - np.pi)
        return angles
    
    def get_middle_angles(self, hand_data):

        angles = {}
        # Calculate angles for the index finger joints
        angles['distal'] = -1.0 * (self.__calculate_angle(hand_data[9], hand_data[10], hand_data[12]) - np.pi)
        angles['proximal'] = -1.0 * (self.__calculate_angle(hand_data[0], hand_data[9], hand_data[10]) - np.pi)

        return angles
    
    def get_thumb_angles(self, hand_data):
        
        angles = {}
        # Calculate angles for the index finger joints
        palm_face = self.__define_plane(hand_data[0], hand_data[5], hand_data[17])
        actuator_offset = np.deg2rad(15)
        actuator_gain = 2.0
        raw_thumb_actuator_angle = self.__calculate_angle(hand_data[0], hand_data[2], hand_data[3]) - np.pi + actuator_offset
        # raw_thumb_actuator_angle = self.__calculate_angle_from_vectors(hand_data[9],hand_data[0], hand_data[2], hand_data[3]) - np.pi + actuator_offset
        angles['actuators'] = actuator_gain * self.__project_angle_onto_plane(raw_thumb_actuator_angle, palm_face[0])
        
        angles['proximal'] = (self.__calculate_angle(hand_data[1], hand_data[2], hand_data[3]) - np.pi)
        angles['distal'] = (self.__calculate_angle(hand_data[2], hand_data[3], hand_data[4]) - np.pi)

        return angles
    
    def get_angles(self, hand_data):
        angles = [0] * 8
        index_angles = self.get_index_angles(hand_data)
        middle_angles = self.get_middle_angles(hand_data)
        thumb_angles = self.get_thumb_angles(hand_data)

        # angles[0] = 0
        # angles[1] = 0
        # angles[2] = 0
        # angles[3] = 0
        # angles.append(thumb_angles['actuators'])
        # angles.append(thumb_angles['proximal'])
        # angles.append(thumb_angles['distal'])
        # angles[4] = index_angles['proximal']
        # angles[5] = index_angles['distal']
        # angles[6] = middle_angles['proximal']
        # angles[7] = middle_angles['distal']


    
        angles[0] = 0
        angles[3] = thumb_angles['actuators']
        angles[6] = thumb_angles['proximal']
        angles[7] = thumb_angles['distal']
        # angles.append(thumb_angles['actuators'])
        # angles.append(thumb_angles['proximal'])
        # angles.append(thumb_angles['distal'])
        angles[1] = index_angles['proximal']
        angles[4] = index_angles['distal']
        angles[2] = middle_angles['proximal']
        angles[5] = middle_angles['distal']


        return angles