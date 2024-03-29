
from tensorboardX import SummaryWriter
from torchsummary import summary
from datetime     import datetime
from scipy        import stats
from argparse     import ArgumentParser
from network      import My_Net
from load_dataset import SIQADataset

import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch, os, yaml, random


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_indexNum(dataset, config, index, status):
    test_ratio = config['test_ratio']
    train_ratio = config['train_ratio']
    trainindex = index[:int(train_ratio * len(index))]
    testindex = index[int((1 - test_ratio) * len(index)):]
    train_index = [] 
    test_index = [] 

    ref_ids = []
    for line0 in open("./data/" + dataset + "/ref_ids_S.txt", "r"):
        line0 = float(line0[:-1])
        ref_ids.append(line0)
    ref_ids = np.array(ref_ids)

    for i in range(len(ref_ids)):
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                print("Error in splitting data")

    if status == 'train':
        index = train_index
    if status == 'test':
        index = test_index

    return len(index)

if __name__ == '__main__':
    # Training settings
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dataset", type=str, default="Waterloo_1")
    parser.add_argument("--weight_decay", type=float, default=0.001)

    args = parser.parse_args()
    
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # logger for tracking experiments
    
    if os.path.exists('/results/performance_logs.txt'):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    ensure_dir('results')
    f = open('./results/performance_logs.txt', 'a+') 
    
    
    #sys.stdout = f
    now = datetime.now()
    print("#==> Experiment date and time = ", now)
    f.write("\n \n #============================== Experiment date and time = %s.==============================#" % now)
    f.write("\n dataset = {:s} epochs = {:d}, batch_size = {:d}, lr = {:f}, weight_decay= {:f}".format(args.dataset, 
                                                                                                       args.epochs, 
                                                                                                       args.batch_size, 
                                                                                                       args.lr, 
                                                                                                       args.weight_decay))
    f.write("\n %s" % config)
    
    #seed = random.randint(10000000, 99999999)   
    seed = 32707809
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("#==> Seed:", seed)
    f.write("\n Seed : {:d}".format(seed))

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
        random.seed(seed)
        random.shuffle(index)
        ensure_dir('model/Waterloo_1')
        save_model = "./model/Waterloo_1/best_W1.pth" 
        model_dir = "./model/Waterloo_1/"
        
    if args.dataset == "Waterloo_2":
        print("#==> Dataset: Waterloo_2")
        index = list(range(1, 11))
        random.seed(seed)
        random.shuffle(index)
        ensure_dir('model/Waterloo_2')
        save_model = "./model/Waterloo_2/best_W2.pth" 
        model_dir = "./model/Waterloo_2/"
        
    print('#==> Random indexes', index)
    
    #os.rmdir('./visualize')
    ensure_dir('visualize/tensorboard')
    writer = SummaryWriter('visualize/tensorboard')

    dataset = args.dataset
    testnum = get_indexNum(dataset, config, index, "test")
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

    Q_index = 0 # 0 for Global Quality score, 1 left Quality, 2 right Quality, 3 stereo Quality.

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
            optimizer.zero_grad()

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

        # test phase
        model.eval()
        y_pred = np.zeros(testnum)
        y_pred_stereo = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L, L_stereo = 0, 0    # L is for the global quality predictions and L_stereo is for the stereo quality predictions

        with torch.no_grad():
            for i, (patchesL,patchesR, (label, label_L, label_R)) in enumerate(test_loader):
                patchesL = patchesL.to(device)
                patchesR = patchesR.to(device)
                label = label.to(device)
                label_L = label_L.to(device)
                label_R = label_R.to(device)
                y_test[i] = label.item()
                outputs = model(patchesL,patchesR)[Q_index]
                #outputs_stereo = model(patchesL,patchesR)[Q_index+3]
                
                score = outputs.mean()
                #score_stereo = outputs_stereo.mean()
                y_pred[i] = score
                #y_pred_stereo[i] = score_stereo

                loss = criterion(score, label[0])
                #loss_stereo = criterion(score_stereo, label[0])
                L = L + loss.item()
                #L_stereo = L_stereo + loss.item()

        test_loss = L / (i + 1)
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())
        
        #test_loss_stereo = L_stereo / (i + 1)
        #SROCC_stereo = stats.spearmanr(y_pred_stereo, y_test)[0]
        #PLCC_stereo = stats.pearsonr(y_pred_stereo, y_test)[0]
        #KROCC_stereo = stats.stats.kendalltau(y_pred_stereo, y_test)[0]
        #RMSE_stereo = np.sqrt(((y_pred_stereo - y_test) ** 2).mean())


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
            plt.scatter(y_pred, y_test, c ="blue")
            plt.xlabel("Prediction")
            plt.ylabel("MOS")
            plt.savefig('mos_vs_pred.png')
            plt.clf()
            #plt.show()

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
        f.write("%s\n" % "Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                                SROCC,
                                                                                                                PLCC,
                                                                                                                KROCC,
                                                                                                                RMSE))
        print("Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                                SROCC,
                                                                                                                PLCC,
                                                                                                                KROCC,
                                                                                                                RMSE))
        f.close() 