import numpy as np


def bilinear_resize(image, new_height: int, new_width: int) -> list:
    """
    Resize an image using bilinear interpolation.

    Args:
        image: 2D (grayscale) or 3D (RGB) array representing an image
        new_height: Target height of the resized image
        new_width: Target width of the resized image

    Returns:
        Resized image as a nested list with values rounded to 2 decimal places
    """
    arr = np.asarray(image, dtype=float)

    if arr.ndim != 2 and arr.ndim != 3:
        return -1

    
    is_gray = arr.ndim == 2
    if is_gray:
        arr = arr[:, :, np.newaxis]

    H, W = arr.shape[0], arr.shape[1]
    scale_y = H / new_height
    scale_x = W / new_width

    out = np.zeros((new_height, new_width, arr.shape[2]))

    for y in range(new_height):
        
        # 1. Ou tombe cette ligne dans l'image d'origine ?
        src_y = y * scale_y
        # 2. Voisins du haut et du bas (on bloque au bord)
        y0 = min(int(np.floor(src_y)), H - 1)
        y1 = min(y0 + 1, H - 1)
        # 3. Distance depuis le voisin du haut (entre 0 et 1)
        dy = src_y - y0

        for x in range(new_width):

            # Meme chose pour les colonnes
            src_x = x * scale_x
            x0 = min(int(np.floor(src_x)), W - 1)
            x1 = min(x0 + 1, W - 1)
            dx = src_x - x0

            # Les 4 voisins (en RGB, chacun est un vecteur [R, G, B])
            A = arr[y0, x0] 
            B = arr[y0, x1]  
            C = arr[y1, x0]  
            D = arr[y1, x1] 

            # 4. Melange horizontal
            haut = A * (1 - dx) + B * dx
            bas = C * (1 - dx) + D * dx

            # 5. Melange vertical
            out[y, x] = haut * (1 - dy) + bas * dy

    if is_gray:
        out = out[:, :, 0]

    return np.round(out, 2).tolist()
