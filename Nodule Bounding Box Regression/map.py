#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from collections import Counter
from iou import intersection_over_union


# In[3]:


def mean_average_precision(
pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20
):
    # pred_boxes (list): [[train_idx, class_preds, prob_score, x1, y1, x2, y2], ...]
    
    average_precisions = []
    epsilon = 1e-6
    
    for c in range(num_classes):
        detections = []
        ground_truths = []
        
        for detection in pred_boxes: # detection variable created
            if detection[1] == c: # 'c' denotes the "class" 
                detections.append(detection) # pushing compared box to the Detection variable
        for true_box in true_boxes:  # true_box varaible assigned
            if true_box[1] == c:
                ground_truths.append(true_box) # pushing the GT_box to truth_box var.
                
        # img 0 has 3 boxes
        # img 1 has 5 boxes
        # amount_bboxes = {0:3, 1:5} -----> {element_key_number_stored, key_value_stored}
        
        amount_bboxes = Counter([gt[0] for gt in ground_truths]) 
        # count how many BB and creates a dictionary
        
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val) 
            # here key will be 0 and 1 
            # torch.zeros() creates a tensor [all values = 0], size = given_argument.
            # this above code updates the key_value for the key_number in  []
            # amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
        
        detections.sort(key=lambda x: x[2], reverse=True) # 2 corresponds to the probab_score!
        # score_probabilities are the values at the 2nd index!
        
        TP = torch.zeros((len(detections))) # true positives
        FP = torch.zeros((len(detections))) # false positives
        total_true_bboxes = len(ground_truths) 
        
        for detection_idx, detection in enumerate(detections):
            #taking out a particular bb for a particular class
            # enumarte() adds a counter to the iterable object and keeps the track of iterations indexes
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
                #taking out the ground truths that has the same index as our detected bbox has
            ]
            
            num_gts = len(ground_truth_img)
            best_iou = 0
            
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                        torch.tensor(detection[3:]),
                        box_format=box_format,
                        # as for IoU we require only coordinates!
                        )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1
        # [1,1,0,1,0] -> [1,2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls)) # takes (y, x)
    return sum(average_precisions) / len(average_precisions)

        
                    #taking out single bb for a class from a image, and taking all gt box for thet image...
#                     now comparing that bb with all the target bb and cheching IOU and taking tracking of that 
#                     best iou....and checking best iou >< iou threshold... means that prediction is correct
                 
    


# In[ ]:




