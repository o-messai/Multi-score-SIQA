import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
from pathlib import Path

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def img_loader(path):
    return Image.open(path).convert('RGB')

def CropPatches(image, patch_size=32, stride=32):
    w, h = image.size
    #image.show()
    patches = ()
    for i in range(0, h-patch_size, stride):
        for j in range(0, w-patch_size, stride):
            patch = to_tensor(image.crop((j, i, j+patch_size, i+patch_size)))
            patches = patches + (patch,)
    return patches

def Split_left_right(image):
    w, h = image.size
    imL = image.crop((0, 0, w/2, h))
    imR = image.crop((w/2, 0, w, h))
    return imL, imR

class SIQADataset(Dataset):
    def __init__(self, dataset, config, index, status):
        self.img_loader = img_loader
        im_dirR = config[dataset]['im_dirR']
        im_dirL = config[dataset]['im_dirL']
        self.patch_size = config['patch_size']
        self.stride = config['stride']

        test_ratio = config['test_ratio']
        train_ratio = config['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, test_index = [], []
        
        im_names = []
        ref_names = []
        for line1 in open("./data/im_names_S.txt", "r"):
            line1 = line1.strip()
            im_names.append(line1)
        im_names = np.array(im_names)
        #print("im_names {}".format (im_names))
        typpe = []
        for line10  in open("./data/im_names_S.txt", "r"):
            line10 = line10.strip()
            line10 = line10.split("/")
            
            typpe.append(line10)
        typpe =[item[0] for item in typpe]
        typpe = np.array(typpe) 
        #print("type {}".format (typpe))

        for line2 in open("./data/refnames_S.txt", "r"):
            line2 = line2.strip()
            ref_names.append(line2)
        ref_names = np.array(ref_names)
        ref_ids = []
        for line0 in open("./data/ref_ids_S.txt", "r"):
            line0 = float(line0[:-1])
            ref_ids.append(line0)
        ref_ids = np.array(ref_ids)

        for i in range(len(ref_ids)):
            if(ref_ids[i] in trainindex):
                train_index.append(i)
            elif(ref_ids[i] in testindex):
                test_index.append(i)
       


        if status == 'train':
            self.index = train_index
            print("#==> Train Images len: {}".format(len(self.index)))
            print('#==> Training Indexes: ', trainindex)
        if status == 'test':
            self.index = test_index
            print("#==> Test Images len: {}".format(len(self.index)))
            print('#==> Test Indexes: ', testindex)
            print('#==> Test Image names: ', im_names[test_index])
            
   

        self.mos_s = []
        self.mos_l = []
        self.mos_r = []

        for line_s in open("./data/MOS_S.txt", "r"):
            line_s = float(line_s.strip())
            self.mos_s.append(line_s)
        self.mos_s = np.array(self.mos_s)
    
        for line_l in open("./data/MOS_L.txt", "r"):
            line_l = float(line_l.strip())
            self.mos_l.append(line_l)
        self.mos_l = np.array(self.mos_l)

        for line_r in open("./data/MOS_R.txt", "r"):
            line_r = float(line_r.strip())
            self.mos_r.append(line_r)
        self.mos_r = np.array(self.mos_r)


        self.patchesR = ()
        self.patchesL = ()
   
        self.label = []
        self.label_L = []
        self.label_R = []
        

        self.im_names = [im_names[i] for i in self.index]
        self.ref_names = [ref_names[i] for i in self.index]
        self.mos_s = [self.mos_s[i] for i in self.index]
        self.mos_l = [self.mos_l[i] for i in self.index]
        self.mos_r = [self.mos_r[i] for i in self.index]

        typpe = [typpe[i] for i in self.index]


        for idx in range(len(self.index)):
         
            img = self.img_loader(Path(str(im_dirL)+ str(self.im_names[idx]) ))
            imL, imR = Split_left_right(img)
         

            patchesR = CropPatches(imR, self.patch_size, self.stride)
            patchesL = CropPatches(imL, self.patch_size, self.stride)

            if status == 'train':
                self.patchesL = self.patchesL + patchesL
                self.patchesR = self.patchesR + patchesR
                
                for i in range(len(patchesL)):
                    self.label.append(self.mos_s[idx])
                    self.label_L.append(self.mos_l[idx])
                    self.label_R.append(self.mos_r[idx])
                    

            elif status == 'test':
                self.patchesL = self.patchesL + (torch.stack(patchesL), )
                self.patchesR = self.patchesR + (torch.stack(patchesR), )
                self.label.append(self.mos_s[idx])
                self.label_L.append(self.mos_l[idx])
                self.label_R.append(self.mos_r[idx])
                

    def __len__(self):
        return len(self.patchesL)

    def __getitem__(self, idx):
        return self.patchesL[idx], self.patchesR[idx], (torch.Tensor([self.label[idx]]), torch.Tensor([self.label_L[idx]]), torch.Tensor([self.label_R[idx]]))
