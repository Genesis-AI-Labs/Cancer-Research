#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch
from iou import intersection_over_union


# In[7]:


def non_max_supression(
            predictions,
            iou_threshold,
            threshold,
            box_format="corners"
):
    # predictions = [ [1, 0.7, x1, x2, y1, y2], [same], [same] ]
    assert type(bboxes) == list
    
    bboxes = [box for box in bboxes if box[1] > threshold] # comparing the class  probabilities.
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    #reverse (highest probability the first), key= lambda x:x[1] sorts using the second element of the array
    # so that choosing the box with the highest class_prob the first! key = comparing_argument
    bboxes_after_nms = []
    while bboxes:
        chosen_box = bboxes.pop(0) 
        # chosing the one with the highest score.
        #The pop() method removes the item at the given index from the list and returns the removed item.
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0] # here 0 denotes the class. ARE THEY OF THE SAME CLASS?
            # I_o_U method defined previously takes 2 arguments  i_o_u(predicted_box, ground_truth_box)
            # ignoring the class number and class_probabs...IoU requires only (x,y) as inputs!
            or intersection_over_union(
                torch.tensor(chosen_box[2:]), # removes the first 2 elements in the predictions list
                #predictions = [ [1, 0.7, x1, x2, y1, y2], [same], [same] ] || 2 and forward [2:]
                # calculation of  IoU using the coordinates from the prediction list only.
                torch.tensor(box[2:]),
                box_format=box_format, # default box format
            )
            < iou_threshold
        ]
        
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms
    


# In[ ]:




