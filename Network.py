import torch
import torch.nn as nn
import torch.nn.functional as F


class My_Net(nn.Module):
    def __init__(self):
        super(My_Net, self).__init__()
        # conv of left view
        self.conv1L = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1) 
        self.conv2L = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3L = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4L = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5L = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # conv of right view
        self.conv1R = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2R = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3R = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4R = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5R = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # conv of stereo view
        self.conv1S = nn.Conv2d(in_channels=6, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2S = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3S = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4S = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5S = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # FC of letf view
        self.bn1_L = nn.BatchNorm2d(32)
        self.bn2_L = nn.BatchNorm2d(32)
        self.bn3_L = nn.BatchNorm2d(128)
        self.fc1_L = nn.Linear(2048, 1024)
        self.fc2_L = nn.Linear(1024, 512)
        self.fc3_L = nn.Linear(512, 1)

        # FC of right view
        self.bn1_R = nn.BatchNorm2d(32)
        self.bn2_R = nn.BatchNorm2d(32)
        self.bn3_R = nn.BatchNorm2d(128)
        self.fc1_R = nn.Linear(2048, 1024)
        self.fc2_R = nn.Linear(1024, 512)
        self.fc3_R = nn.Linear(512, 1)

        # FC of stereo
        self.bn1_S = nn.BatchNorm2d(32)
        self.bn2_S = nn.BatchNorm2d(32)
        self.bn3_S = nn.BatchNorm2d(128)
        self.fc1_S = nn.Linear(2048, 1024)
        self.fc2_S = nn.Linear(1024, 512)
        self.fc3_S = nn.Linear(512, 1)

        # FC of global score
        self.fc1_1 = nn.Linear(512 * 3, 1024)  
        self.fc2_1 = nn.Linear(1024, 512)
        self.fc3_1 = nn.Linear(512, 1)
        
    def forward(self, xL,xR):

        x_distort_L = xL.view(-1, xL.size(-3), xL.size(-2), xL.size(-1))
        x_distort_R = xR.view(-1, xR.size(-3), xR.size(-2), xR.size(-1))

     ####################################################### left view  ####################################################  
        
        Out_L = F.relu(self.conv1L(x_distort_L)) 
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)#32x16×16
        Out_L = self.bn1_L(Out_L)

        Out_L = F.relu(self.conv2L(Out_L))#32x8×8
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)
        Out_L = self.bn2_L(Out_L)


        Out_L = F.relu(self.conv3L(Out_L))#64x8×8
        Out_L = F.relu(self.conv4L(Out_L))#64x8×8

        Out_L = F.relu(self.conv5L(Out_L))
        Out_L = F.max_pool2d(Out_L, (2, 2), stride=2)#128x4×4
        Out_L = self.bn3_L(Out_L)

        
        Out_L = Out_L.view(-1, self.num_flat_features(Out_L))
        Out_L = self.fc1_L(Out_L)#512
        Out_L = F.dropout(Out_L, p=0.5, training=True, inplace=False)

        Out_LF = self.fc2_L(Out_L)#512
        #Out_LF = F.dropout(Out_L, p=0.0, training=True, inplace=False)

        Out_L = F.relu(self.fc3_L(Out_LF))
                    
     ####################################################### right view  ####################################################  
        
        
        Out_R = F.relu(self.conv1R(x_distort_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn1_R(Out_R)

        Out_R = F.relu(self.conv2R(Out_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn2_R(Out_R)
      
        Out_R = F.relu(self.conv3R(Out_R))
        Out_R = F.relu(self.conv4R(Out_R))
        

        Out_R = F.relu(self.conv5R(Out_R))
        Out_R = F.max_pool2d(Out_R, (2, 2), stride=2)
        Out_R = self.bn3_R(Out_R)
        
        Out_R = Out_R.view(-1, self.num_flat_features(Out_R))
        Out_R = self.fc1_R(Out_R)
        Out_R = F.dropout(Out_R, p=0.5, training=True, inplace=False)

        Out_RF = self.fc2_R(Out_R)
        #Out_R = F.dropout(Out_R, p=0.0, training=True, inplace=False)
        
        Out_R = F.relu(self.fc3_R(Out_RF))
                      
     ####################################################### Stereo view  ####################################################  
        
        input_s = torch.cat((x_distort_L, x_distort_R), dim=1)

        Out_S = F.relu(self.conv1S(input_s))
        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn1_S(Out_S)

        Out_S = F.relu(self.conv2S(Out_S))
        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn2_S(Out_S)
      
        Out_S = F.relu(self.conv3S(Out_S))
        Out_S = F.relu(self.conv4S(Out_S))
        

        Out_S = F.relu(self.conv5S(Out_S))
        Out_S = F.max_pool2d(Out_S, (2, 2), stride=2)
        Out_S = self.bn3_S(Out_S)
        
        Out_S = Out_S.view(-1, self.num_flat_features(Out_S))
        Out_S = self.fc1_S(Out_S)
        Out_S = F.dropout(Out_S, p=0.5, training=True, inplace=False)

        Out_SF = self.fc2_S(Out_S)
        #Out_S = F.dropout(Out_S, p=0.0, training=True, inplace=False)

        Out_S = F.relu(self.fc3_S(Out_SF))


        #############################################    Concatenation  ############################

        cat_total = F.relu(torch.cat((Out_RF, Out_LF, Out_SF), dim=1))
        #fc_total = cat_total.view(-1, self.num_flat_features(cat_total))
        ################################## Quality score prediction  ##########################################

        Quality_left = Out_L
        Quality_right = Out_R
        Quality_stereo = Out_S 

        fc1_1 = self.fc1_1(cat_total)
        fc2_1 = F.relu(self.fc2_1(fc1_1))
        Global_Quality = F.relu(self.fc3_1(fc2_1))                        # Global Quality score prediction

        return Global_Quality, Quality_left, Quality_right, Quality_stereo


    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



