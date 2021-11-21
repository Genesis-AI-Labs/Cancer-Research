#!/usr/bin/env python
# coding: utf-8

# ## Loading Libraries

# In[1]:


import os
import glob
import copy
import time
import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from collections import namedtuple
import SimpleITK as sitk

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


# ## Explore Dataset

# In[2]:


df_annotations = pd.read_csv('LUNA/annotations.csv')
df_annotations.head()


# In[3]:


df_annotations.shape


# In[4]:


df_candidates = pd.read_csv('LUNA/candidates.csv')
df_candidates.head()


# In[5]:


df_candidates.shape


# In[6]:


# there are multiple annotations and candidates of a single CT scan

print(f'Total Annotations: {df_annotations.shape[0]}, Unique CT scans: {len(df_annotations.seriesuid.unique())}')
print(f'Total Candidates: {df_candidates.shape[0]}, Unique CT scans: {len(df_candidates.seriesuid.unique())}')


# In[7]:


diameters = {}

for _, row in df_annotations.iterrows():
    center_xyz = (row.coordX, row.coordY, row.coordZ)
    
    diameters.setdefault(row.seriesuid, []).append(
        (center_xyz, row.diameter_mm)
    )


# In[8]:


len(diameters)


# Combining candidates and annotations

# In[9]:


get_ipython().run_cell_magic('time', '', "\nCandidateInfoTuple = namedtuple(\n    'CandidateInfoTuple',\n    ['is_nodule', 'diameter_mm', 'series_uid', 'center_xyz']\n)\n\ncandidates = []\n\nfor _, row in df_candidates.iterrows():\n\n    candidate_center_xyz = (row.coordX, row.coordY, row.coordZ)\n\n    candidate_diameter = 0.0\n\n    for annotation in diameters.get(row.seriesuid, []):\n\n        annotation_center_xyz, annotation_diameter = annotation\n\n        for i in range(3):\n\n            delta = abs(candidate_center_xyz[i] - annotation_center_xyz[i])\n\n            if delta > annotation_diameter / 4:\n                    break\n            \n\n        else:\n            candidate_diameter = annotation_diameter\n            \n\n            break\n    \n    candidates.append(CandidateInfoTuple(\n        bool(row['class']),\n        candidate_diameter,\n        row.seriesuid,\n        candidate_center_xyz\n    ))")


# In[10]:


candidates.sort(reverse=True)


# In[11]:


with open('LUNA/missing.txt', 'r') as f:
    missing_uids = {uid.split('\n')[0] for uid in f}

len(missing_uids)


# In[12]:


candidates_clean = list(filter(lambda x: x.series_uid not in missing_uids, candidates))

print(f'All candidates in dataset: {len(candidates)}')
print(f'Candidates with CT scan : {len(candidates_clean)}')


# ## Loading the Data

# In[13]:


candidate = candidates_clean[0]

candidate


# In[14]:


filepaths = glob.glob(f'LUNA/subset*/*/{candidate.series_uid}.mhd')

#glob can return multiple files with specified patterns
#sanity check for discarded files

assert len(filepaths) != 0, f'CT scan with seriesuid {candidate.series_uid} not found!'

filepaths


# In[15]:


mhd_file_path = filepaths[0]

mhd_file_path


# In[16]:


# Reading the Image using SimpleITK

mhd_file = sitk.ReadImage(mhd_file_path)


# In[17]:


#store it as numpy array

ct_scan = np.array(sitk.GetArrayFromImage(mhd_file), dtype = np.float32)


# In[18]:


ct_scan.clip(-1000, 1000, ct_scan)


# In[19]:


origin_xyz = mhd_file.GetOrigin()
voxel_size_xyz = mhd_file.GetSpacing()
direction_matrix = np.array(mhd_file.GetDirection()).reshape(3, 3)


# In[20]:


origin_xyz_np = np.array(origin_xyz)
voxel_size_xyz_np = np.array(voxel_size_xyz)


# In[21]:


# Patient Coordinate System --> CRI Voxel Coordinate System

cri = ((center_xyz - origin_xyz_np) @ np.linalg.inv(direction_matrix)) / voxel_size_xyz_np

cri  = np.round(cri)


# CRI --> IRC

irc = (int(cri[2]), int(cri[1]), int(cri[0]))


# In[22]:


ct_scan.shape


# In[23]:


# extract a chunk of size 10 along the index column, and 18 rows and columns.

dims_irc = (10, 18, 18)


# In[24]:



slice_list = []

for axis, center_val in enumerate(irc):
    

    start_index = int(round(center_val - dims_irc[axis]/2))
    end_index = int(start_index + dims_irc[axis])


    if start_index < 0:
        start_index = 0
        end_index = int(dims_irc[axis])
    

    if end_index > ct_scan.shape[axis]:
        end_index = ct_scan.shape[axis]
        start_index = int(ct_scan.shape[axis] - dims_irc[axis])
        
    slice_list.append(slice(start_index, end_index))
    
tuple(slice_list)


# In[25]:


ct_scan_chunk = ct_scan[tuple(slice_list)]
ct_scan_chunk.shape


# In[26]:


candidate.is_nodule


# In[27]:


# is_nodule tensor

torch.tensor([
    not candidate.is_nodule,
    candidate.is_nodule,
], dtype = torch.long)


# In[28]:


# convert ct scan chunk ---> Pytorch tensor

# The code in this cell is from the Deep Learning with PyTorch book's GitHub repository
# https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/util/disk.py

# The imports have slightly been modified to make the code work


import gzip

from cassandra.cqltypes import BytesType
from diskcache import FanoutCache, Disk, core
from diskcache.core import io, MODE_BINARY
from io import BytesIO

class GzipDisk(Disk):
    def store(self, value, read, key=None):

        # pylint: disable=unidiomatic-typecheck
        if type(value) is BytesType:
            if read:
                value = value.read()
                read = False

            str_io = BytesIO()
            gz_file = gzip.GzipFile(mode='wb', compresslevel=1, fileobj=str_io)

            for offset in range(0, len(value), 2**30):
                gz_file.write(value[offset:offset+2**30])
            gz_file.close()

            value = str_io.getvalue()

        return super(GzipDisk, self).store(value, read)


    def fetch(self, mode, filename, value, read):

        value = super(GzipDisk, self).fetch(mode, filename, value, read)

        if mode == MODE_BINARY:
            str_io = BytesIO(value)
            gz_file = gzip.GzipFile(mode='rb', fileobj=str_io)
            read_csio = BytesIO()

            while True:
                uncompressed_data = gz_file.read(2**30)
                if uncompressed_data:
                    read_csio.write(uncompressed_data)
                else:
                    break

            value = read_csio.getvalue()

        return value

def getCache(scope_str):
    return FanoutCache('data-unversioned/cache/' + scope_str,
                       disk=GzipDisk,
                       shards=64,
                       timeout=1,
                       size_limit=3e11,
                       )

raw_cache = getCache('ct_scan_raw')

@raw_cache.memoize(typed=True)
def getCtScanChunk(series_uid, center_xyz, dims_irc):

        filepaths = glob.glob(f'LUNA/subset*/*/{series_uid}.mhd')
        assert len(filepaths) != 0, f'CT scan with seriesuid {series_uid} not found!'
        mhd_file_path = filepaths[0]
        
        mhd_file = sitk.ReadImage(mhd_file_path)
        ct_scan = np.array(sitk.GetArrayFromImage(mhd_file), dtype=np.float32)
        ct_scan.clip(-1000, 1000, ct_scan)
        
        origin_xyz = mhd_file.GetOrigin()
        voxel_size_xyz = mhd_file.GetSpacing()
        direction_matrix = np.array(mhd_file.GetDirection()).reshape(3, 3)
        
        origin_xyz_np = np.array(origin_xyz)
        voxel_size_xyz_np = np.array(voxel_size_xyz)
        
        cri = ((center_xyz - origin_xyz_np) @ np.linalg.inv(direction_matrix)) / voxel_size_xyz_np
        cri = np.round(cri)
        irc = (int(cri[2]), int(cri[1]), int(cri[0]))
        
        slice_list = []
        for axis, center_val in enumerate(irc):
            
            start_index = int(round(center_val - dims_irc[axis]/2))
            end_index = int(start_index + dims_irc[axis])
            
            if start_index < 0:
                start_index = 0
                end_index = int(dims_irc[axis])
                
            if end_index > ct_scan.shape[axis]:
                end_index = ct_scan.shape[axis]
                start_index = int(ct_scan.shape[axis] - dims_irc[axis])

            slice_list.append(slice(start_index, end_index))
            
        ct_scan_chunk = ct_scan[tuple(slice_list)]
        
        return ct_scan_chunk


# In[29]:


# Creating PyTorch Dataset

class LunaDataset(Dataset):
    def __init__(self, is_validation_set = False, validation_stride = 0):
        self.candidates = copy.copy(candidates_clean[::350])
        
        if is_validation_set:
            self.candidates = self.candidates[::validation_stride]
            
        else:
            del self.candidates[::validation_stride]
            
    def __len__(self):
        return len(self.candidates)
    
    def __getitem__(self, i):
        candidate = self.candidates[i]
        dims_irc = (10, 18, 18)
        ct_scan_np = getCtScanChunk(candidate.series_uid, candidate.center_xyz, dims_irc)
        
        ct_scan_tensor = torch.from_numpy(ct_scan_np).to(torch.float32).unsqueeze(0)
        
        label_tensor = torch.tensor([
            not candidate.is_nodule,
            candidate.is_nodule
        ], dtype=torch.long)
        
        return ct_scan_tensor, label_tensor
        


# In[30]:


VALIDATION_STRIDE = 10 # every 10th CT Scan will be in Validation dataset
BS=16

train_ds = LunaDataset(is_validation_set = False, validation_stride =  VALIDATION_STRIDE)
val_ds = LunaDataset(is_validation_set = True, validation_stride = VALIDATION_STRIDE)

train_dl = DataLoader(train_ds, batch_size=BS, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BS, num_workers=0)


# In[43]:


def train_loop(model, dataloader, criterion, optimizer, ds_size):


    model.train()

    running_loss = 0.0
    running_corrects = 0

    running_pos = 0
    running_pos_correct = 0

    running_neg = 0
    running_neg_correct = 0
    
    for inputs, labels in tqdm(dataloader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        loss = criterion(outputs, labels[:,1])
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data[:,1])
        

        running_pos += labels.data[:,1].sum()
        running_pos_correct += ((preds == labels.data[:,1]) & (labels.data[:,1] == 1)).sum()
        

        running_neg += labels.data[:,0].sum()
        running_neg_correct += ((preds == labels.data[:,1]) & (labels.data[:,1] == 0)).sum()

    epoch_loss = running_loss / ds_size
    epoch_acc = running_corrects.double() / ds_size
    
    return epoch_loss, epoch_acc, (running_pos_correct, running_pos), (running_neg_correct, running_neg)
    
    

def eval_loop(model, dataloader, criterion, ds_size):



    model.eval()


    running_loss = 0.0
    running_corrects = 0
    
    running_pos = 0
    running_pos_correct = 0
    running_neg = 0
    running_neg_correct = 0
    

    with torch.no_grad():
    
        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
        
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels[:,1])
        
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data[:,1])
            
            running_pos += labels.data[:,1].sum()
            running_pos_correct += ((preds == labels.data[:,1]) & (labels.data[:,1] == 1)).sum()

            running_neg += labels.data[:,0].sum()
            running_neg_correct += ((preds == labels.data[:,1]) & (labels.data[:,1] == 0)).sum()
        
    epoch_loss = running_loss / ds_size
    epoch_acc = running_corrects.double() / ds_size
    
    return epoch_loss, epoch_acc, (running_pos_correct, running_pos), (running_neg_correct, running_neg)


# ## Train the Model

# In[44]:


class LunaModel(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1, bias =True)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(2)
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(2048, 1024)
        self.relu3 = nn.ReLU()
        
        self.dropout = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(1024, 2)
        
    def forward(self, X):
        
        X = self.maxpool1(self.relu1(self.conv1(X)))
        X = self.maxpool2(self.relu2(self.conv2(X)))
        
        X = self.flatten(X)
        
        X = self.relu3(self.fc1(X))
        X = self.dropout(X)
        
        return self.fc2(X)


# In[45]:


model = LunaModel()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), weight_decay = 0.1)


# In[46]:


EPOCHS = 5

for epoch in range(EPOCHS):

    epoch_start = time.time()

    train_loss, train_acc, train_pos, train_neg = train_loop(
        model, train_dl, criterion,
        optimizer, len(train_ds)
    )

    val_loss, val_acc, val_pos, val_neg = eval_loop(
        model, val_dl, criterion, len(val_ds)
    )

    time_elapsed = time.time() - epoch_start
    print(f'Epoch: {epoch+1:02} | Epoch Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print()
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\tTrain - correct pos: {train_pos[0]}/{train_pos[1]} | correct neg: {train_neg[0]}/{train_neg[1]}')
    print()
    print(f'\tVal. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%')
    print(f'\tVal. - correct pos: {val_pos[0]}/{val_pos[1]} | correct neg: {val_neg[0]}/{val_neg[1]}')
    print()


# In[ ]:




