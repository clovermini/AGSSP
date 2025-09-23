import torch
import os, sys
o_path = os.getcwd()
sys.path.append(o_path)
sys.path.append(os.path.join(o_path, '../'))

from PIL import Image
import json
from collections import OrderedDict


def initialize_memory(obj_list):

    mid = []
    large = []
    patch = []
    for x in obj_list:
        mid.append((x, []))
        large.append((x, []))
        patch.append((x, []))
    mid_memory   = OrderedDict(mid)
    large_memory = OrderedDict(large)
    patch_memory = OrderedDict(patch)
    return mid_memory, large_memory, patch_memory


@torch.no_grad()
def memory_surgery(model, obj_list, des_path, preprocess, k_shot, device, feature_list, dpam_layer, ignore_residual=False):
    
    with open(des_path) as f:
        des=json.load(f)
    
    miss_obj = set()
    for obj in obj_list:
        samples = des[obj]['samples']
        if len(samples) < 1:
            miss_obj.add(obj)
            continue
    print('miss_obj ', miss_obj)
    obj_list = [x for x in obj_list if x not in miss_obj]

    mem_features = {}
    for obj in obj_list:
        samples = des[obj]['samples']  # [:k_shot]
        print('good samples ', len(samples))

        features = []

        for image in samples:
            img = Image.open(os.path.join(image))
		    # transforms
            if preprocess is not None:
                img = preprocess(img)
            images = img.to(device).unsqueeze(0)
            cls_name = [obj]
            with torch.no_grad():
                class_tokens, _, patch_tokens = model.encode_image(images, feature_list, dpam_layer, ignore_residual)
                patch_tokens = [p[:, 1:, :] for p in patch_tokens]
                features.append(patch_tokens)
        mem_features[obj] = [torch.cat(
            [features[j][i] for j in range(len(features))], dim=0) for i in range(len(features[0]))]
    return mem_features
