import os
import sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(os.path.join(o_path, '../'))

import time
import json
import torch
import torch.nn as nn
import random
import argparse
import numpy as np

import torch.nn.functional as F
import torchvision.transforms as transforms
from models import open_clip
from few_shot import memory_surgery
from dataset import datasets
from utils import visualizer
from metrics import metrics
from tqdm import tqdm
from logging import getLogger

# from open_clip import get_tokenizer   # tokenizer
import warnings
import argparse


def setup_seed(seed):  # 设置随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class prompt_order():  # prompt 函数
    def __init__(self, des_path) -> None:
        super().__init__()

        with open(des_path) as f:
            des=json.load(f)
        self.total_des = des

        self.state_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
        ]

        self.state_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage",
        ]

        self.template_list =[
        "a cropped photo of the {}.",
        "a close-up photo of a {}.",
        "a close-up photo of the {}.",
        "a bright photo of a {}.",
        "a bright photo of the {}.",
        "a dark photo of the {}.",
        "a dark photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of the {}.",
        "a blurry photo of a {}.",
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of the {} for visual inspection.",
        "a photo of a {} for visual inspection.",
        "a photo of the {} for anomaly detection.",
        "a photo of a {} for anomaly detection."
        ]


    def prompt(self, class_name, use_detailed=True):
        des_info = self.total_des[class_name]
        print('class_name ', class_name)
        class_name = des_info['map']
        # class_name = 'metal surface'
        print('map class_name ', class_name)

        state_normal_list = self.state_normal_list.copy()
        state_anomaly_list = self.state_anomaly_list.copy()
        
        if use_detailed:
            state_normal_list.extend(des_info['des']['good'])
            state_anomaly_list.extend(des_info['des']['defect'])
        
        print('state_normal_list ', state_normal_list)
        print('state_anomaly_list ', state_anomaly_list)

        class_state = [ele.format(class_name) for ele in state_normal_list]
        normal_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]
    
        class_state = [ele.format(class_name) for ele in state_anomaly_list]
        anomaly_ensemble_template = [class_template.format(ele) for ele in class_state for class_template in self.template_list]

        empty_template = [class_template.format('') for class_template in self.template_list]
        return normal_ensemble_template, anomaly_ensemble_template, empty_template


class CLIP_AD(nn.Module):
    def __init__(self, model_name = 'ViT-B-16-plus-240', pretrain = 'laion400m_e32', img_size=240):
        super(CLIP_AD, self).__init__()

        self.model, _, self.preprocess = open_clip.create_customer_model_and_transforms(model_name, pretrained=pretrain, force_image_size=img_size)

        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
    
    def encode_text(self, text):

        text = self.tokenizer(text, context_length=self.model.context_length)
        text_token, _ = self.model.encode_text(text.cuda())
        text_token /= text_token.norm(dim=-1, keepdim=True)  
        return text_token
    
    def encode_image(self, image, feature_list=None, DPAM_layer=None, ignore_residual=False):   # 图像编码

        class_tokens, tokens, patch_tokens = self.model.encode_image(image, None, proj = True, feature_list = feature_list, DPAM_layer = DPAM_layer, ignore_residual = ignore_residual)  # feature_list = [3, 6, 9, 12], DPAM_layer = 10

        return class_tokens, tokens, patch_tokens


def compute_score(image_features, text_features):  # 计算 图像文本相似得分
    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    text_probs = (torch.bmm(image_features.unsqueeze(1), text_features)/0.07).softmax(dim=-1)

    return text_probs


def compute_sim(image_features, text_features):  # 计算 图像文本相似度
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)
    simmarity = (torch.bmm(image_features.squeeze(2), text_features)/0.07).softmax(dim=-1)
    return simmarity


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    #sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm


def few_shot(memory, token, class_name, idx):
    retrive = []
    for i in class_name:
        L, N, D = memory[i][idx].shape   # [980, 1, 640]    [5, 225, 640]
        retrive.append(memory[i][idx].permute(2, 1, 0).reshape(D,-1)) # D NL   # [640, 225*5]
    retrive = torch.stack(retrive)# B D NL   # [32, 640, 225*5]   
    # print('retrive ', retrive.shape, ' token ', token.shape)
    # B D L    [32, 169, 640]  [32, 640, 225*5]   
    M = 1/2 * torch.min(1.0 - torch.bmm(F.normalize(token.squeeze(2), dim = -1), F.normalize(retrive, dim = 1)), dim = -1)[0]
    return M


def prepare_text_feature(model, obj_list, des_path, use_detailed=True):  # 准备文本特征   obj_list: cate list
    Mermory_avg_normal_text_features = []
    Mermory_avg_abnormal_text_features = []
    Mem_redundant_features = []
    text_generator = prompt_order(des_path)

    for obj_name in obj_list:

        normal_description, abnormal_description, empty_template = text_generator.prompt(obj_name, use_detailed=use_detailed)  # alu

        normal_text_features = model.encode_text(normal_description).float()
        abnormal_text_features = model.encode_text(abnormal_description).float()
        empty_features = model.encode_text(empty_template).float()

        avg_normal_text_features = torch.mean(normal_text_features, dim = 0, keepdim= True) 
        avg_normal_text_features /= avg_normal_text_features.norm()
        avg_abnormal_text_features = torch.mean(abnormal_text_features, dim = 0, keepdim= True)   # 取平均
        avg_abnormal_text_features /= avg_abnormal_text_features.norm()
        redundant_features = torch.mean(empty_features, dim = 0, keepdim= True)   # 取平均
        redundant_features /= redundant_features.norm()

        Mermory_avg_normal_text_features.append(avg_normal_text_features)
        Mermory_avg_abnormal_text_features.append(avg_abnormal_text_features)
        Mem_redundant_features.append(redundant_features)
    Mermory_avg_normal_text_features = torch.stack(Mermory_avg_normal_text_features)      # [2, 1, 640]  
    Mermory_avg_abnormal_text_features = torch.stack(Mermory_avg_abnormal_text_features)  # [2, 1, 640]  
    Mem_redundant_features = torch.stack(Mem_redundant_features)
    print('Mermory_avg_normal_text_features ', Mermory_avg_normal_text_features.shape) 
    print('Mermory_avg_abnormal_text_features ', Mermory_avg_abnormal_text_features.shape)   
    print('Mem_redundant_features ', Mem_redundant_features.shape)
    return Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features, Mem_redundant_features


def set_logger(txt_path):
    import logging
     # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a+')  # w
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


@torch.no_grad()
def test(args,):
    img_size = args.image_size
    patch_size = args.patch_size   # 14  # 16
    feature_list = args.feature_list   # [3,6,9]
    dpam_layer = args.dpam_layer
    if dpam_layer < 1:
        dpam_layer = None
    print('feature_list ', feature_list, ' dpam_layer ', dpam_layer)
    dataset_dir = args.data_path
    des_path = args.des_path
    meta_path = args.meta_path
    save_path = args.save_path
    dataset_name = args.dataset
    k_shot = args.k_shot
    surgery_type = args.surgery_type
    if '_res' in surgery_type:
        ignore_residual = False   # args.ignore_residual
        surgery_type = surgery_type.replace('_res', '')
    else:  # ignore_residual 
        ignore_residual = True
    print('surgery_type ', surgery_type, ' ignore_residual ', ignore_residual)
    visualize = args.visualize
    save_anomaly_map = args.save_anomaly_map
    use_detailed = args.use_detailed

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, 'log.txt')
    logger = set_logger(txt_path)

    print('**************** args ***************')
    for k,v in sorted(vars(args).items()):
        logger.info("%s", str(k)+' = '+str(v))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIP_AD(args.model, args.pretrained, img_size=img_size)   # model_name
    model.to(device)
    model.model.visual.DAPM_replace(DPAM_layer = dpam_layer, surgery_type=surgery_type)   # clip surgery

    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),   # img_size 240
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    
    preprocess = model.preprocess

    preprocess.transforms[0] = transforms.Resize(size=(img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC,
                                                 max_size=None, antialias=None)
    preprocess.transforms[1] = transforms.CenterCrop(size=(img_size, img_size))

    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
        test_data = datasets.MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test', obj_name=obj_list)
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
        test_data = datasets.VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test', obj_name=obj_list)
    elif dataset_name == 'metal_own':

        obj_list = datasets.CLSNAMES
        test_data = datasets.MetalDataset(root=dataset_dir, meta_path=meta_path, transform=preprocess, target_transform=transform, mode='test', k_shot=k_shot, save_dir=save_path, obj_name=obj_list)
        print('******* running ... obj_list ', obj_list)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)  # 32

    model.eval()
    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []  # image level text_probs
    results['pr_sp'] = [] # image level label
    
    Mermory_avg_normal_text_features, Mermory_avg_abnormal_text_features, Mem_redundant_features = prepare_text_feature(model, obj_list, des_path, use_detailed)
    ########################################
    if k_shot == 0:
        few = False
    else:
        few = True
    
    print('############ few_shot ', few, ' k_shot ', args.k_shot)
    if few:
        mem_features = memory_surgery(model.to(device), obj_list, des_path, preprocess, args.k_shot, device, feature_list, dpam_layer, ignore_residual)

    st_time = time.time()  
    for index, items  in enumerate(tqdm(test_dataloader)):
        images = items['img'].to(device)
        cls_name = items['cls_name']

        cls_id = items['cls_id']
        results['cls_names'].extend(cls_name)
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].extend(items['anomaly'].detach().cpu())

        b, c, h, w = images.shape   # [32, 3, 240, 240]
  
        average_normal_features = Mermory_avg_normal_text_features[cls_id]
        average_anomaly_features = Mermory_avg_abnormal_text_features[cls_id]
        redundant_features = Mem_redundant_features[cls_id]
        
        text_features = torch.cat((average_normal_features - redundant_features, average_anomaly_features - redundant_features), dim = 1)

        image_features, tokens, patch_features = model.encode_image(images, feature_list, dpam_layer, ignore_residual)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        text_probs = compute_score(image_features, text_features.permute(0, 2, 1))   # [32, 1, 2]
        text_probs = text_probs[:, 0, 1]  # z0score 

        anomaly_map_list = []
        for idx, patch_feature in enumerate(patch_features):
            patch_feature = patch_feature/ patch_feature.norm(dim = -1, keepdim = True)   # [32, 226, 640]

            similarity = compute_sim(patch_feature, text_features.permute(0, 2, 1)) # [:,:,1]   # [32, 169]
            similarity_map = get_similarity_map(similarity[:, 1:, :], args.image_size)
            anomaly_map = similarity_map[...,1]
            anomaly_map_list.append(anomaly_map)

        anomaly_map = torch.stack(anomaly_map_list)
        anomaly_map = anomaly_map.mean(dim = 0)
        
        if few:  # few-shot
            anomaly_maps_few_shot = []
            for idx, p in enumerate(patch_features):
                p = p[:, 1:, :]  # 去除 cls

                cos = few_shot(mem_features, p, cls_name, idx)
                anomaly_map_few_shot  =  cos.reshape((b, h//patch_size, w//patch_size)).cuda()
                anomaly_maps_few_shot.append(anomaly_map_few_shot.cpu().numpy())

            anomaly_map_few_shot = np.mean(anomaly_maps_few_shot, axis=0)
            anomaly_map_few_shot = torch.from_numpy(anomaly_map_few_shot)
            anomaly_map = (anomaly_map.cpu() + anomaly_map_few_shot) 

            text_probs = (text_probs.cpu() + torch.max(torch.max(anomaly_map, dim = 1)[0],dim = 1)[0])/2.0   # [32]
        
        anomaly_map_final = F.interpolate(torch.tensor(anomaly_map).unsqueeze(1), size=img_size, mode='bilinear', align_corners=True)
        anomaly_map_final = anomaly_map_final.squeeze(1)
        results['pr_sp'].extend(text_probs.detach().cpu())
        results['anomaly_maps'].append(anomaly_map_final)

        # 可视化
        if visualize and k_shot == 4:
            show_path = os.path.join(save_path, 'vis/')  #
            if not os.path.exists(show_path):
                os.mkdir(show_path)
            visualizer.vis(items['img_path'], anomaly_map_final, text_probs.detach().cpu(), img_size, show_path, items['cls_name'], gt_mask.squeeze(1))

        if save_anomaly_map and k_shot == 4:
  
            anomaly_map = anomaly_map.cpu().numpy()
            for idx, path in enumerate(items['img_path']):
                cls_name = items['cls_name'][idx]
                image_name = path.split('/')[-1]
                path = os.path.join(save_path, 'anomaly_map/', cls_name)
                if not os.path.exists(path):
                    os.makedirs(path)
                anomaly_save_path = os.path.join(path, image_name.split('.')[0]+'_anomaly.npy')

                ano_map = anomaly_map[idx]
                np.save(anomaly_save_path, ano_map)
                print('saving ... anomaly_save_path ', anomaly_save_path)

    results['imgs_masks'] = torch.cat(results['imgs_masks'])
    results['anomaly_maps'] = torch.cat(results['anomaly_maps']).detach().cpu().numpy()
    print('anomaly_maps ', results['anomaly_maps'].shape, ' imgs_masks ', results['imgs_masks'].shape)
    print('get anomaly_maps ', time.time()-st_time, ' for len ', len(results['anomaly_maps']))

    st_time = time.time()
    metric_results = metrics.cal_metrics(obj_list, results)
    logger.info("\n%s", metric_results)
    print('cal_metrics costs ', time.time()-st_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/test', help='path to save results')
    parser.add_argument("--des_path", type=str, default='', help='path to defect description')
    parser.add_argument("--meta_path", type=str, default='', help='path to data')
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--feature_list", type=int, nargs="+", default=[3, 6, 9, 12], help="features used")   # [3, 6, 9, 12], DPAM_layer = 10
    parser.add_argument("--dpam_layer", type=int, default=10, help="surgery layer")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--patch_size", type=int, default=16, help="image size")
    # parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    # few shot
    parser.add_argument("--k_shot", type=int, default=10, help="10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")

    parser.add_argument("--surgery_type", type=str, default="vv", help="clip surgery/clearclip")
    parser.add_argument("--use_detailed", action='store_true')
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--save_anomaly_map", action='store_true')

    args = parser.parse_args()

    setup_seed(args.seed)
    test(args)
