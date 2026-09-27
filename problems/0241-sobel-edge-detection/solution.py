import numpy as np

def sobel_edge_detection(image):
    """
    Apply Sobel edge detection to a grayscale image.

    Args:
        image: 2D list/array representing a grayscale image
               with values in range [0, 255]

    Returns:
        Edge magnitude image as 2D list with integer values (0-255),
        or -1 if input is invalid
    """
    arr = np.asarray(image, dtype=float)

    if arr.ndim != 2 or arr.size == 0:
        return -1

    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return -1

    if np.any(arr < 0) or np.any(arr > 255):
        return -1

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    height, width = arr.shape
    output = np.zeros((height - 2, width - 2))        # <-- CHANGEMENT 1

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx, gy = 0, 0

            for ki in range(3):
                for kj in range(3):
                    pixel = arr[i + ki - 1, j + kj - 1]
                    gx += pixel * Gx[ki, kj]
                    gy += pixel * Gy[ki, kj]

            magnitude = np.sqrt(gx**2 + gy**2)
            output[i - 1, j - 1] = magnitude          

    max_val = output.max()
    if max_val > 0:
        output = output / max_val * 255

    return np.round(output).astype(int).tolist()