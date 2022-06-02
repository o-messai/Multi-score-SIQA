
from torchsummary import summary
import torch
import os
import numpy as np
from scipy import stats
import yaml
from argparse import ArgumentParser
import random
import torch.nn as nn
from Network import My_Net
from Load_dataset import SIQADataset


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_indexNum(config, index, status):
    test_ratio = config['test_ratio']
    train_ratio = config['train_ratio']
    trainindex = index[:int(train_ratio * len(index))]
    testindex = index[int((1 - test_ratio) * len(index)):]
    train_index = [] 
    val_index = []
    test_index = [] 

    ref_ids = []
    for line0 in open("./data/ref_ids_S.txt", "r"):
        line0 = float(line0[:-1])
        ref_ids.append(line0)
    ref_ids = np.array(ref_ids)

    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)

    if status == 'train':
        index = train_index
    if status == 'test':
        index = test_index
    if status == 'val':
        index = val_index

    return len(index)

if __name__ == '__main__':
    # Training settings
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dataset", type=str, default="Waterloo_1")
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    args = parser.parse_args()
    
    seed = random.randint(10000000, 99999999) 
    #seed = 32707809
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("#==> Seed:", seed)

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device != 'cpu':
        print('#==> Using GPU device:', torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('#==> Using CPU....')

    index = []
    if args.dataset == "Waterloo_1":
        print("#==> Dataset: Waterloo_1")
        index = list(range(1, 7))
        random.shuffle(index)
    print('#==> Random indexes', index)
    

    ensure_dir('results')
    save_model = "./results/model_W1.pth" 
    model_dir = "./results/"
 

    dataset = args.dataset
   
    testnum = get_indexNum(config, index, "test")

    train_dataset = SIQADataset(dataset, config, index, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)


    test_dataset = SIQADataset(dataset, config, index, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset)


    ###model
    model = My_Net().to(device)
    summary(model, input_size = [(3, 32, 32), (3, 32, 32)] )

    Q_index = 0
    ###
    criterion = nn.L1Loss() #nn.L1Loss(size_average=None, reduce=True, reduction= 'mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.1, last_epoch=-1)
    ###
    best_PLCC = -1
    best_RMSE = 100000
    # training phase
    for epoch in range(args.epochs):
        
        model.train()
        LOSS_all = 0
        LOSS_G , LOSS_L, LOSS_R, LOSS_S = 0, 0, 0, 0

        for i, (patchesL, patchesR,(label, label_L, label_R)) in enumerate(train_loader):
            patchesL = patchesL.to(device)
            patchesR = patchesR.to(device)
            label = label.to(device)
            label_L = label_L.to(device)
            label_R = label_R.to(device)

            optimizer.zero_grad()

            global_Quality = model(patchesL,patchesR)[0]
            Quality_left   = model(patchesL,patchesR)[1]
            Quality_right  = model(patchesL,patchesR)[2]
            Quality_stereo = model(patchesL,patchesR)[3]

            loss_G = criterion(global_Quality, label)
            loss_L = criterion(Quality_left,   label_L)
            loss_R = criterion(Quality_right,  label_R)
            loss_S = criterion(Quality_stereo, label)

            Lamda, sigma, delta, theta = 2, 1, 1, 1
            loss = ((Lamda * loss_G) + (sigma * loss_L) + (delta * loss_R) + (theta * loss_S))

            loss.backward()
            optimizer.step()

            LOSS_all = LOSS_all + loss.item()
            LOSS_G = LOSS_G + loss_G.item()
            LOSS_L = LOSS_L + loss_L.item()
            LOSS_R = LOSS_R + loss_R.item()
            LOSS_S = LOSS_S + loss_S.item()
          
        train_loss_all = LOSS_all / (i + 1)
        print ('#==> Training_loss : ',train_loss_all)

        train_loss_g = LOSS_G / (i + 1)
        #print('#==> Global Quality score training loss',train_loss_g)

        train_loss_L = LOSS_L / (i + 1)
        #print('#==> Left Quality score training loss',train_loss_L)

        train_loss_R = LOSS_R / (i + 1)
        #print('#==> Right Quality score training loss',train_loss_R)

        train_loss_S = LOSS_S / (i + 1)
        #print('#==> Stereo Quality score training loss',train_loss_S)

        
        model.eval()
        #L = 0
   

        # test phase
        y_pred = np.zeros(testnum)
        y_pred_stereo = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L, L_stereo = 0, 0

        with torch.no_grad():
            for i, (patchesL,patchesR, (label, label_L, label_R)) in enumerate(test_loader):
                patchesL = patchesL.to(device)
                patchesR = patchesR.to(device)
                label = label.to(device)
                label_L = label_L.to(device)
                label_R = label_R.to(device)

                y_test[i] = label.item()
 
                

                outputs = model(patchesL,patchesR)[Q_index]
                outputs_stereo = model(patchesL,patchesR)[Q_index+3]
                
                score = outputs.mean()
                score_stereo = outputs_stereo.mean()
                y_pred[i] = score
                y_pred_stereo[i] = score_stereo

                loss = criterion(score, label[0])
                loss_stereo = criterion(score_stereo, label[0])
                L = L + loss.item()
                L_stereo = L_stereo + loss.item()

        test_loss = L / (i + 1)
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())

        test_loss_stereo = L_stereo / (i + 1)
        SROCC_stereo = stats.spearmanr(y_pred_stereo, y_test)[0]
        PLCC_stereo = stats.pearsonr(y_pred_stereo, y_test)[0]
        KROCC_stereo = stats.stats.kendalltau(y_pred_stereo, y_test)[0]
        RMSE_stereo = np.sqrt(((y_pred_stereo - y_test) ** 2).mean())


        print("#==> Epoch {} Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                            test_loss,
                                                                                                            SROCC,
                                                                                                            PLCC,
                                                                                                            KROCC,
                                                                                                            RMSE))


        if RMSE < best_RMSE :
            print("#==> Update Epoch {} best valid RMSE".format(epoch))
            torch.save(model.state_dict(), os.path.join(model_dir, 'epoch-{}.pth'.format(epoch)))
            torch.save(model.state_dict(), save_model)
            #best_PLCC = PLCC
            best_RMSE = RMSE

    ########################################################################## final test ############################################
    model.load_state_dict(torch.load(save_model))
    model.eval()
    with torch.no_grad():
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)

        L = 0
    
        for i, (patchesL,patchesR, (label, label_L, label_R)) in enumerate(test_loader):
 
            patchesL = patchesL.to(device)
            patchesR = patchesR.to(device)
            label = label.to(device)
            label_L = label_L.to(device)
            label_R = label_R.to(device)

            y_test[i] = label.item()

         
            outputs = model(patchesL,patchesR)[Q_index]


            score = outputs.mean()
            y_pred[i] = score

           
    #################################################### SROCC/PLCC/KROCC/RMSE score ####################################################
    SROCC = stats.spearmanr(y_pred, y_test)[0]
    PLCC = stats.pearsonr(y_pred, y_test)[0]
    KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
    RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())

    
    if os.path.exists('total_result.txt'):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    with open('total_result.txt', 'a+') as f:
        f.seek(1)
        f.write("%s\n" % "Waterloo 1 : Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                                SROCC,
                                                                                                                PLCC,
                                                                                                                KROCC,
                                                                                                                RMSE))
        print("Phase 1 : Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                                SROCC,
                                                                                                                PLCC,
                                                                                                                KROCC,
                                                                                                                RMSE))

        f.close() 
    