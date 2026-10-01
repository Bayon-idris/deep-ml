import numpy as np

def pca_color_augmentation(image: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """
    Apply PCA color augmentation to an RGB image.

    Args:
        image: RGB image of shape (H, W, 3) with values in [0, 255]
        alpha: Array of 3 random coefficients for principal components

    Returns:
        Augmented image of shape (H, W, 3) with values clamped to [0, 255]
    """
    num_channels = image.shape[2]
    image_flatten = image.reshape(-1, num_channels).astype(float)

    cov_matrix = np.cov(image_flatten, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = np.maximum(eigenvalues, 0)

    sign_refs = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=float)
    for k in range(num_channels):
        if np.dot(eigenvectors[:, k], sign_refs[k]) < 0:
            eigenvectors[:, k] = -eigenvectors[:, k]

    perturbation = eigenvectors @ (alpha * np.sqrt(eigenvalues))

    augmented_image_flatten = image_flatten + perturbation
    augmented_image_clipped = np.clip(augmented_image_flatten, 0, 255)

    return augmented_image_clipped.reshape(image.shape)