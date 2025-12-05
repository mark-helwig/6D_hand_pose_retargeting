import open3d as o3d

"""
Class for visualizing the hand landmarks and camera using open 3D
"""

class Vis3D():
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.mesh_frame)

        # Initialize left and right hand point clouds
        self.pcd_left = o3d.geometry.PointCloud()
        self.pcd_left.paint_uniform_color([1, 0.706, 0])
        self.vis.add_geometry(self.pcd_left)

        self.pcd_right = o3d.geometry.PointCloud()
        self.pcd_right.paint_uniform_color([0, 0.651, 0.929])
        self.vis.add_geometry(self.pcd_right)

        self.pcd_hand = o3d.geometry.PointCloud()
        self.pcd_hand.paint_uniform_color([0, 0.651, 0.929])
        self.vis.add_geometry(self.pcd_hand)

        self.blue = [0, 0, 1]
        self.red = [1, 0, 0]


    def show_hand(self,data,color=[0, 0, 1]):
        # return none if no data
        if data.shape != (21,3):
            return None
        else:
            self.vis.remove_geometry(self.pcd_hand)
            self.pcd_hand = o3d.geometry.PointCloud()
            self.pcd_hand.points = o3d.utility.Vector3dVector(data)
            self.pcd_hand.paint_uniform_color(color)
            self.vis.add_geometry(self.pcd_hand)

              # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()

    def visualize_angle_projection(self, origin, vec_a, vec_b, plane_origin=None, plane_normal=None,
                                   radius=0.05, color=[0.8, 0.2, 0.2], vector_color=[0, 1, 0],
                                   proj_color=[1, 0, 0], n_points=64):
        """Project two vectors (from `origin`) onto a plane and visualize the projection and angle.

        Parameters
        - origin: (3,) array-like, the common origin point of the two vectors.
        - vec_a, vec_b: (3,) array-like, direction vectors (not necessarily unit) starting at origin.
        - plane_origin: (3,) array-like, a point on the plane. If None, uses `origin`.
        - plane_normal: (3,) array-like, normal vector of the plane. If None, uses Z axis [0,0,1].
        - radius: float, radius of the arc that visualizes the angle on the plane.
        - color: arc color.
        - vector_color: color for the original vectors.
        - proj_color: color for the projected vectors.
        - n_points: resolution of the arc.
        """
        import numpy as _np

        o = _np.asarray(origin, dtype=float)
        a = _np.asarray(vec_a, dtype=float)
        b = _np.asarray(vec_b, dtype=float)

        if plane_origin is None:
            plane_origin = o
        else:
            plane_origin = _np.asarray(plane_origin, dtype=float)

        if plane_normal is None:
            n = _np.array([0.0, 0.0, 1.0])
        else:
            n = _np.asarray(plane_normal, dtype=float)

        # normalize plane normal
        n = n / _np.linalg.norm(n)

        # project vectors onto plane: v_proj = v - (v·n) n
        a_proj = a - _np.dot(a, n) * n
        b_proj = b - _np.dot(b, n) * n

        # If projection near-zero, nothing meaningful to show
        if _np.linalg.norm(a_proj) < 1e-8 or _np.linalg.norm(b_proj) < 1e-8:
            return

        ua = a_proj / _np.linalg.norm(a_proj)
        ub = b_proj / _np.linalg.norm(b_proj)

        # Build orthonormal basis (u, v) on the plane: u = ua normalized, v = cross(n, u)
        u = ua
        v = _np.cross(n, u)
        v = v / _np.linalg.norm(v)

        # Compute start/end angles in the plane basis
        def angle_in_basis(vec):
            x = _np.dot(vec, u)
            y = _np.dot(vec, v)
            return _np.arctan2(y, x)

        th_a = angle_in_basis(ua)
        th_b = angle_in_basis(ub)

        # Choose shortest angle direction, maintain sign per plane normal
        # Create arc angles linearly between th_a and th_b
        # Ensure arc goes the shorter way
        dth = th_b - th_a
        # wrap to [-pi, pi]
        dth = (dth + _np.pi) % (2 * _np.pi) - _np.pi
        thetas = _np.linspace(th_a, th_a + dth, max(3, int(_np.abs(dth) / (2 * _np.pi) * n_points)))
        if thetas.size < 3:
            thetas = _np.linspace(th_a, th_b, n_points)

        arc_pts = [plane_origin + radius * (_np.cos(t) * u + _np.sin(t) * v) for t in thetas]
        arc_pts = _np.asarray(arc_pts)

        # Build LineSet geometries
        geometries = []

        # Original vectors (from origin to origin + vector)
        p_orig = _np.vstack([o, o + a])
        lines_orig = [[0, 1]]
        colors_orig = [vector_color]
        ls_orig = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(p_orig),
                                       lines=o3d.utility.Vector2iVector(lines_orig))
        ls_orig.colors = o3d.utility.Vector3dVector(colors_orig)
        geometries.append(ls_orig)

        # Projected vectors
        p_proj_a = _np.vstack([o, o + a_proj])
        p_proj_b = _np.vstack([o, o + b_proj])
        ls_proj = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(_np.vstack([p_proj_a, p_proj_b])),
                                       lines=o3d.utility.Vector2iVector([[0, 1], [2, 3]]))
        ls_proj.colors = o3d.utility.Vector3dVector([proj_color, proj_color])
        geometries.append(ls_proj)

        # Arc LineSet
        arc_lines = [[i, i + 1] for i in range(len(arc_pts) - 1)]
        ls_arc = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(arc_pts),
                                      lines=o3d.utility.Vector2iVector(arc_lines))
        ls_arc.colors = o3d.utility.Vector3dVector([color for _ in arc_lines])
        geometries.append(ls_arc)

        # Remove previous angle geoms if present
        if hasattr(self, '_angle_geoms'):
            for g in getattr(self, '_angle_geoms'):
                try:
                    self.vis.remove_geometry(g)
                except:
                    pass

        self._angle_geoms = geometries

        for g in geometries:
            self.vis.add_geometry(g)

        # Update visualization
        self.vis.poll_events()
        self.vis.update_renderer()

