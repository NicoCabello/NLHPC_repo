import torch
from torch.utils.data import Dataset
import numpy as np

from utils import bicubic_interpolation

class ImageDataset(Dataset):
    def __init__(self, X_dir, y_dir, max_conc):
        super(ImageDataset, self).__init__()
        self.X = np.array( [bicubic_interpolation(x,factor=2) for x in np.load(X_dir)] )
        self.y = np.load(y_dir)
        
        self.max_conc = max_conc

    def __getitem__(self, idx):
        return  torch.Tensor( np.expand_dims( self.X[idx] / self.max_conc, 0) ),\
                torch.Tensor( np.expand_dims( self.y[idx] / self.max_conc , 0) )   # 1 channel in axis 0
    
    def __len__(self):
        return len(self.X)

# Create a dataset for evaluation
class EvalDataset(Dataset):
    def __init__(self, X_dir, y_dir, max_conc):
        super(EvalDataset, self).__init__()
        self.X = np.array( [bicubic_interpolation(x,factor=2) for x in np.load(X_dir)] )
        self.y = np.load(y_dir)
        
        self.max_conc = max_conc

    def __getitem__(self, idx):
        return torch.Tensor( np.expand_dims( self.X[idx] / self.max_conc, 0) ),\
               torch.Tensor( np.expand_dims( self.y[idx] / self.max_conc , 0) )   # 1 channel in axis 0
    
    def __len__(self):
        return len(self.X)
    
# Create a dataset for testing. Includes lon/lat information
class TestDataset(Dataset):
    def __init__(self, X_dir, y_dir, lat_lims_dir, lon_lims_dir, max_conc):
        super(TestDataset, self).__init__()
        self.X_low = np.load(X_dir)
        self.X = np.array( [bicubic_interpolation(x,factor=2) for x in self.X_low] )
        self.y = np.load(y_dir)
        self.lat_lims = np.load(lat_lims_dir)
        self.lon_lims = np.load(lon_lims_dir)
        self.max_conc = max_conc

    def __getitem__(self, idx):
        return torch.Tensor( np.expand_dims( self.X[idx] / self.max_conc, 0) ),\
               np.array( self.X_low[idx] ), \
               torch.Tensor( np.expand_dims( self.y[idx] / self.max_conc , 0) ), \
               (self.lat_lims[idx][0], self.lat_lims[idx][1], int(self.lat_lims[idx][2])), \
               (self.lon_lims[idx][0], self.lon_lims[idx][1], int(self.lon_lims[idx][2]))
    
    def __len__(self):
        return len(self.X)