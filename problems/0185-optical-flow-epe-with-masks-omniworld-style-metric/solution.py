from typing import Optional, Union

try:
    import numpy as np
except Exception:
    np = None

ArrayLike = Union[list, "np.ndarray"]

def flow_epe(pred: ArrayLike,
             gt: ArrayLike,
             mask: Optional[ArrayLike] = None,
             max_flow: Optional[float] = None) -> float:
    """
    Compute mean End-Point Error (EPE) between predicted and ground-truth optical flow.

    Args:
        pred, gt: (H, W, 2) lists or NumPy arrays.
        mask: optional (H, W) or broadcastable to (H, W); 1=include, 0=ignore.
        max_flow: optional float; clip per-pixel EPE to this value.

    Returns:
        float: mean EPE over valid pixels. Returns -1 on invalid input or if no valid pixels.
    """
    # Your implementation here


    epe  = 0
    pred = np.asarray(pred)
    gt = np.asarray(gt) 

    if pred.ndim != 3 or pred.shape[2] !=2 :
        return -1

    if gt.shape != pred.shape:
        return -1    

    diff = pred - gt
    epe  = np.sqrt(np.sum(diff ** 2, axis=2))    
    valid = np.isfinite(epe)

    if mask is not None:
        mask = np.asarray(mask)
        valid &= (mask != 0)

    if max_flow is not None:
        epe = np.minimum(epe, max_flow)

    valid_epe = epe[valid]   
    if valid_epe.size == 0:
        return -1

    return (np.mean(valid_epe,))         

