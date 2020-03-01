from sklearn.metrics import f1_score, roc_auc_score

import numpy as np

# At least one dict key should match name of function

def iou(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou[0]


def _two_by_two(y_true, _y_pred, class_threshold, iou_threshold):
    '''
    Calculates TP/FP/TN/FN for 1 class for 1 image 
    @ 1 classification score threshold and 1 IoU threshold.

    y_pred : numpy array of shape (num_pred, 5)
             y_pred[:,-1] = predicted classification scores
    y_true : numpy array of shape (num_true, 4)

    '''
    y_pred = _y_pred.copy()
    y_pred = y_pred[y_pred[:,-1] >= class_threshold, :-1]
    assert y_pred.shape[-1] == 4
    # For negative images ...
    if y_true.shape[0] == 0:
        # If no boxes in truth/predictions, return -1 (for later filtering)
        if y_pred.shape[0] == 0: 
            return -1
        else:
            return 0
    else:
        # If no predictions, then ...
        # TP=0, FP=0, FN=num_true
        if y_pred.shape[0] == 0:
            return 0, 0, len(y_true)
    # For positive images ...
    tp = 0 ; fp = 0 ; fn = 0
    # Loop through ground truth boxes
    for ind, gt in enumerate(y_true):
        # If there are no predictions remaining,
        # assign FNs if gt boxes remain
        if len(y_pred) == 0: 
            fn += len(y_true) - ind
            break
        # Compute IoU of each ground truth box against all predictions
        # First, expand gt box 
        repeat_gt = np.repeat(np.expand_dims(gt, axis=0), len(y_pred), axis=0)
        iou_vals = iou(repeat_gt, y_pred)
        # Threshold based on IoU 
        accepted = iou_vals >= iou_threshold
        # If none meet IoU threshold, gt box is FN
        if np.sum(accepted) == 0:
            fn += 1
        else:
            tp += 1
            # Eliminate box with highest IoU from further consideration
            y_pred = np.delete(y_pred, np.argmax(iou_vals), 0)
    # Unmatched predictions are FPs
    fp += len(y_pred)
    return tp, fp, fn


def class_map(y_true, y_pred, class_threshold, iou_threshold):
    map_list = []
    for img in range(len(y_pred)):
        result = _two_by_two(y_true[img], y_pred[img], class_threshold, iou_threshold)
        if result in (0,-1):
            map_list.append(result)
        else:
            tp, fp, fn = result
            map_list.append(float(tp)/(tp+fp+fn))
    return map_list


def overall_map(y_true, y_pred, class_thresholds, iou_thresholds):
    num_classes = len(y_pred[0])
    # Each element in this list will contain a dictionary corresponding
    # to each class. Each dictionary will have key:value pairs, 
    # where key is the threshold and value is the corresponding mAP.
    per_class_results_list = []
    for each_class in range(num_classes):
        threshold_results_dict = {}
        for class_thresh in class_thresholds:
            # Extract predictions and labels for that class
            class_preds = [pred[each_class] for pred in y_pred]
            # Add 1 to each_class because 0 represents background
            # for class annotations label format
            gt_labels = [gt['bboxes'][gt['labels'] == each_class+1] for gt in y_true]
            # I can take the mean over the lists
            # Because the negative image results will only be affected
            # by class threshold, not IoU threshold
            iou_results_list = []
            for iou_thresh in iou_thresholds:
                iou_results_list.append(class_map(gt_labels, class_preds, class_thresh, iou_thresh))
            iou_results_array = np.asarray(iou_results_list)
            averaged_over_iou = np.mean(iou_results_array, axis=0)
            averaged_over_img = np.mean(averaged_over_iou[averaged_over_iou != -1])
            threshold_results_dict[class_thresh] = averaged_over_img
        per_class_results_list.append(threshold_results_dict)
    # Now I need to find the max mAP/threshold pair for each class
    results_dict = {}
    overall_map = 0.
    for each_class in range(num_classes):
        threshold_results_dict = per_class_results_list[each_class]
        maps_for_class = [threshold_results_dict[ct] for ct in class_thresholds]
        results_dict['class{}_map'.format(each_class+1)] = np.max(maps_for_class)
        results_dict['class{}_thr'.format(each_class+1)] = class_thresholds[maps_for_class.index(np.max(maps_for_class))]
        overall_map += results_dict['class{}_map'.format(each_class+1)]
    results_dict['overall_map'] = overall_map / num_classes
    return results_dict


def overall_map_75(y_true, y_pred, class_thresholds, **kwargs):
    _results_dict = overall_map(y_true, y_pred, class_thresholds, [0.75])
    results_dict = {}
    for k,v in _results_dict.items():
        results_dict[k+'_75'] = v
    return results_dict


def overall_auc(y_true, y_pred, **kwargs):
    num_classes = len(y_pred[0])
    per_class_results_list = []
    for each_class in range(num_classes):
        class_preds = [pred[each_class] for pred in y_pred]
        class_probs = [np.max(pred[...,-1]) if len(pred) > 0 else 0 for pred in class_preds]
        gt_labels = [1 if gt['bboxes'][gt['labels'] == each_class+1].shape[0] > 0 else 0 for gt in y_true]
        per_class_results_list.append(roc_auc_score(gt_labels, class_probs))
    results_dict = {}
    overall_auc = 0.
    for each_class in range(num_classes):
        results_dict['class{}_auc'.format(each_class+1)] = per_class_results_list[each_class]
        overall_auc += results_dict['class{}_auc'.format(each_class+1)]
    results_dict['overall_auc'] = overall_auc / num_classes
    return results_dict


