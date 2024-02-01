import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import PolypPVT
from utils.dataloader import test_dataset
import cv2
import time
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='/home/ti-2/Desktop/CPSNet/Camouflaged Object Detection/model_pth/Camouflaged/22Camouflaged.pth')
    opt = parser.parse_args()
    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    for _data_name in ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']:

        ##### put data_path here #####
        data_path = './dataset/COD-TestDataset/{}'.format(_data_name)
        ##### save_path #####
        save_path = './result_map/best/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            P1,P2,P3,P4 = model(image)
            res = F.upsample(P1+P2+P3+P4, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)
        # print(_data_name, 'Finish!')
    end_time = time.time()
    times = end_time - start_time
    print(times)
