import numpy as np

def flip_image(image, direction):
    arr = np.asarray(image)

    if arr.ndim not in (2, 3):
        return -1

    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return -1

    if direction.lower() not in ('horizontal', 'vertical'):
        return -1

    height = arr.shape[0]
    width = arr.shape[1]

    flipped_image = []

    for i in range(height):
        row = []

        for j in range(width):
            if direction.lower() == 'horizontal':
                row.append(arr[i][width - 1 - j])
            else:
                row.append(arr[height - 1 - i][j])

        flipped_image.append(row)

    return flipped_image