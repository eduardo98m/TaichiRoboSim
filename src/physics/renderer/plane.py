import numpy as np

def render_plane(normal: np.ndarray, offset: np.ndarray, extent: float = 20.0):
    """
    Render the plane

    Arguments:
    ----------
    normal : np.ndarray
        -> Normal vector of the plane
    offset : np.ndarray
        -> Offset of the plane
    
    extent : float
        -> Extent of the plane (length of each side) // This is for rendering purpose only

    Output:
    -------
    vertices : np.ndarray
        -> Vertices of the plane. It should be an array of shape (4, 3)
    faces    : np.ndarray
        -> Faces of the plane. It should be an array of shape (2, 3)
    """
    
    # Create a rotation matrix that rotates the z-axis to the normal vector
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, normal)
    c = np.dot(z_axis, normal)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rot_mat = np.eye(3) + kmat + kmat.dot(kmat) * (1 / (1 + c))

    # Create the vertices of the plane
    x = np.linspace(-extent / 2, extent / 2, 2)
    y = np.linspace(-extent / 2, extent / 2, 2)
    xv, yv = np.meshgrid(x, y)
    zv = np.zeros_like(xv)
    vertices = np.stack((xv.flatten(), yv.flatten(), zv.flatten()), axis=-1)

    # Rotate and translate the vertices
    vertices = (rot_mat @ vertices.T).T + offset

    # Create the faces of the plane
    faces = np.array([[0, 1, 2], [1, 3, 2]])

    return vertices, faces
