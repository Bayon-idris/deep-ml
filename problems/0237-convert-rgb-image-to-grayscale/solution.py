import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    arr = np.asarray(image)
    luminosity = np.array([0.299, 0.587, 0.114])

    if arr.ndim != 3 or arr.shape[-1] != 3 :
        return -1

    if np.any(arr <0 )  or np.any(arr > 255 ) :
        return -1    

    grayscale_img = np.round(np.dot(arr, luminosity)).astype(int)


    return grayscale_img.tolist()
