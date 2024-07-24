from typing import Union
import numpy as np 
from torch import Tensor


def pixel_to_coords(i, j, Nextra, noise_scale):
    pixel_points = []

    # cartesian coordinates of pixel
    x = i 
    y = j

    # perturb position of point so img doesn't look gridded
    point = np.array([x, y])
    pixel_point = point + np.random.normal(loc=0.0, scale=noise_scale, size=(2,))

    pixel_points.append(pixel_point)

    # add some extra points for each pixel
    for i in range(Nextra):
        pixel_points.append(
            # add a few noisy points around the given point
            point + np.random.normal(loc=0.0, scale=noise_scale, size=(2,))
        )
    return pixel_points


def image_to_point_cloud(
    img: Union[np.ndarray, Tensor],
    n_pix: int = 28,                # size of image
    n_extra: int = 3,               # number of extra points to add
    noise_scale: int = 0.5,        # Units of pixels
    n_points_subsample: int = 400, # Number of total points to subsample
    use_density: bool = True
)-> np.ndarray:
    """ Convert image tensor to point cloud """
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]]) # 90 deg rotation

    if isinstance(img, Tensor):
        img = img.numpy()
    if img.ndim == 3:
        img = img.squeeze() # Remove channel --> numpy array

    points = []  # (x, y) coordinates
    density = [] # pixel values
    for i in range(n_pix):
        for j in range(n_pix):
            # Skip the empty pixels
            if img[i, j] == 0.:
                continue
            else:
                # Make the (x,y) points for each pixel
                points.append(
                    pixel_to_coords(i, j, n_extra, noise_scale)
                )
                # Copy the pixel value accordingly
                density.extend(
                    [img[i, j]] * (n_extra + 1)
                )

    density = np.asarray(density)
    points = np.asarray(points).reshape(-1, 2)
    points = np.matmul(points, rotation) # Rotate to match image

    # Scale points to [-1, 1]
    points = 2. * (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0)) - 1.

    if n_points_subsample is not None:
        idx = np.random.randint(0, len(points), size=(n_points_subsample,))
        points = points[idx]
        density = density[idx]

    if use_density:
        cloud = np.concatenate([points, density[:, np.newaxis]], axis=1)
    else:
        cloud = points
    return cloud