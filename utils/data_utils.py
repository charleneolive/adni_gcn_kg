import cv2
import torch
from scipy.ndimage import zoom
from torch.utils.data import DataLoader,Dataset

class PrepareDataset(Dataset):
    """prepare adni and ppmi mri"""

    def __init__(self, images, transform, dtype=torch.float32):
#images, batch_size=bs, prep=prep, min_size=224, dtype=torch.float32
        self.images = images
        self.transform = transform
        self.dtype = dtype
        self.size = 227

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        
        im = zoom(image, (227/image.shape[0], 227/image.shape[1], 110/image.shape[2]))
            
#         im = cv2.resize(image, dsize=(self.size, self.size))
        if self.transform:
#             im_array = self.prep(im)
            tensor_array = torch.unsqueeze(self.transform(im),0)
            im_array = torch.cat((tensor_array,tensor_array,tensor_array),0)
        # need to handle numerical and categorical variable
        return im_array
    
class PrepareDataset2(Dataset):
    """prepare adni and ppmi mri for sfcn"""

    def __init__(self, images, transform, dtype=torch.float32):
#images, batch_size=bs, prep=prep, min_size=224, dtype=torch.float32
        self.images = images
        self.transform = transform
        self.dtype = dtype
        self.size = 227

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        im = zoom(image, (192/image.shape[0], 160/image.shape[1], 160/image.shape[2]))
        
#         im = cv2.resize(image, dsize=(self.size, self.size))
        if self.transform:
#             im_array = self.prep(im)
            tensor_array = torch.unsqueeze(self.transform(im),0)
        # need to handle numerical and categorical variable
        return tensor_array
    
# class PrepareDataset_Adni(Dataset):
#     """prepare adni mri"""

#     def __init__(self, images, transform, dtype=torch.float32):
# #images, batch_size=bs, prep=prep, min_size=224, dtype=torch.float32
#         self.images = images
#         self.transform = transform
#         self.dtype = dtype
#         self.size = 227

#     def __len__(self):
#         return self.images.shape[0]

#     def __getitem__(self, idx):
#         image = self.images[idx]
        
#         im = cv2.resize(image, (self.size, self.size))
            
# #         im = cv2.resize(image, dsize=(self.size, self.size))
#         if self.transform:
# #             im_array = self.prep(im)
#             tensor_array = torch.unsqueeze(self.transform(im),0)
#             im_array = torch.cat((tensor_array,tensor_array,tensor_array),0)
#         # need to handle numerical and categorical variable
#         return im_array
    
# class PrepareDataset2(Dataset):
#     """prepare adni and ppmi mri, only 20 scans"""

#     def __init__(self, images, transform, dtype=torch.float32):
# #images, batch_size=bs, prep=prep, min_size=224, dtype=torch.float32
#         self.images = images
#         self.transform = transform
#         self.dtype = dtype
#         self.size = 227

#     def __len__(self):
#         return self.images.shape[0]

#     def __getitem__(self, idx):
#         image = self.images[idx]
        
#         im = zoom(image, (227/image.shape[0], 227/image.shape[1], 20/image.shape[2]))
            
# #         im = cv2.resize(image, dsize=(self.size, self.size))
#         if self.transform:
# #             im_array = self.prep(im)
#             tensor_array = torch.unsqueeze(self.transform(im),0)
#             im_array = torch.cat((tensor_array,tensor_array,tensor_array),0)
#         # need to handle numerical and categorical variable
#         return im_array