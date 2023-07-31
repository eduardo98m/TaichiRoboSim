import numpy as np

def render_heightfield(x_cooridnates : np.ndarray, 
                       y_coordinates : np.ndarray, 
                       heightfield   : np.ndarray):
    """
    Render the heightfield

    Arguments:
    ----------
    `x_cooridnates` : np.ndarray
        -> X coordinates of the heightfield given as a 1D array. It hass shape (n_cols,)
    `y_coordinates` : np.ndarray
        -> Y coordinates of the heightfield given as a 1D array. It hass shape (n_rows,)
    `heightfield`   : np.ndarray
        -> Heightfield data given as a 2D array (matrix). It has shape (n_rows, n_cols)

    Output:
    -------
    `vertices` : np.ndarray
        -> Vertices of the heightfield. It should be an array of shape (n_rows * n_cols, 3)
    `faces`    : np.ndarray
        -> Faces of the heightfield. It should be an array of shape (n_rows * n_cols, 3)
    """
    # Get the number of rows and columns in the heightfield
    n_rows, n_cols = heightfield.shape
    
    # Create an array of indices for each vertex in the heightfield
    indices = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    
    # Create an array of vertices for each point in the heightfield
    vertices = np.column_stack((x_cooridnates.repeat(n_rows), 
                                y_coordinates.repeat(n_cols), 
                                heightfield.ravel()))
    
    # Create an array of faces for each square in the heightfield
    faces = np.column_stack((indices[:-1,:-1].ravel(), 
                             indices[:-1,1:].ravel(), 
                             indices[1:,:-1].ravel()))
    
    # Duplicate each face to create two triangles per square
    faces = np.vstack((faces, np.column_stack((indices[1:,1:].ravel(), 
                                                indices[1:,:-1].ravel(), 
                                                indices[:-1,1:].ravel()))))
    
    return vertices, faces
    

