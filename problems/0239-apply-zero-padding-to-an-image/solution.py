import numpy as np 
 
def zero_pad_image(img, pad_width): 
    """ 
    Add zero padding around a grayscale image. 
     
    Args: 
        img: 2D list or numpy array of pixel values 
        pad_width: integer number of pixels to pad on each side 
     
    Returns: 
        Padded image as 2D list with integer values, 
        or -1 if input is invalid 
    """ 

    arr = np.asarray(img) 
 
    if arr.ndim != 2:
        return -1 
 
    if pad_width < 0: 
        return -1 
 
    filled_padd = [] 
    
    height = arr.shape[0] 
    width = arr.shape[1] 

    new_size = width + 2 * pad_width

    for _ in range(pad_width):
        filled_padd.append([0] * new_size)

    for i in range(height):
        row = []

        for _ in range(pad_width):
            row.append(0)

        for j in range(width):
            row.append(arr[i][j])

        for _ in range(pad_width):
            row.append(0)

        filled_padd.append(row)

    for _ in range(pad_width):
        filled_padd.append([0] * new_size)

    return filled_padd