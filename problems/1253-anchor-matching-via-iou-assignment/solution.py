import numpy as np

def bbox_iou (boxA, boxB):

	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    unionArea = boxAArea + boxBArea - interArea
    if unionArea <= 0:
        return 0.0
    iou = interArea / float(unionArea)
    return iou

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign each anchor a training label via IoU matching.

    Args:
        anchors: (N, 4) boxes in xyxy format [x1, y1, x2, y2]
        gt_boxes: (M, 4) ground-truth boxes in xyxy format
        pos_threshold: IoU >= this → positive (default 0.5)
        neg_threshold: IoU <  this → negative (default 0.4)

    Returns:
        labels:     (N,) int array with values {1=pos, 0=neg, -1=ignore}
        matched_gt: (N,) int array of matched GT index, or -1
    """
    anchors_array = np.asarray(anchors , dtype = float)
    gt_boxes_array = np.asarray(gt_boxes , dtype = float)

    N = anchors_array.shape[0]
    M = gt_boxes_array.shape[0]
    if N == 0:
        return np.zeros(0, dtype = int) , np.zeros(0 , dtype = int)  

    if M == 0:
        matched_gt = np.full(N, -1 , dtype = int)
        labels = np.zeros(N, dtype=int)
        return labels , matched_gt

    iou_matrix = np.zeros((N,M) , dtype=float)   
    for i in range(N):
        for j in range(M):
            iou_matrix[i][j] = bbox_iou(anchors_array[i], gt_boxes_array[j])

    best_iou = np.max(iou_matrix, axis=1)
    best_gt = np.argmax(iou_matrix, axis=1)

    labels = np.full(N, -1 , dtype=int)
    matched_gt = np.full(N, -1, dtype = int)

    for i in range(N):
        if best_iou[i] >= pos_threshold:
            labels[i] = 1
            matched_gt[i] = best_gt[i]
        elif best_iou[i] < neg_threshold:
            labels[i] = 0    

    for j in range(M):
        colonne = iou_matrix[:, j]         
        meilleur_anchor = np.argmax(colonne)
        labels[meilleur_anchor] = 1
        matched_gt[meilleur_anchor] = j

    return labels , matched_gt    
