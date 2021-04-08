import argparse
from collections import OrderedDict
import numpy as np
from scipy import misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.cocostuff_loader_ours import *
from data.vg import *
from model.resnet_generator_v1 import *
from utils.util import *
import imageio
from tqdm import tqdm


def get_dataloader(dataset='coco', img_size=128):
    if dataset == 'coco':
        dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/images/val2017/',
                                        instances_json='./datasets/coco/annotations/instances_val2017.json',
                                        stuff_json='./datasets/coco/annotations/stuff_val2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        with open("./datasets/vg/vocab.json", "r") as read_file:
            vocab = json.load(read_file)
        dataset = VgSceneGraphDataset(vocab=vocab,
                                      h5_path='./datasets/vg/val.h5',
                                      image_dir='./datasets/vg/images/',
                                      image_size=(128, 128), left_right_flip=False, max_objects=30)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=1)
    return dataloader


def main(args):
    device = torch.device('cuda')
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31
    if args.dataset == 'coco':
        background_cls = 92
        foreground_cls = 91

    dataloader = get_dataloader(args.dataset, img_size=64)

    #netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).cuda()
    netG = background_foreground_generator(background_cla=background_cls, foreground_cla=foreground_cls, output_dim=3)

    # if not os.path.isfile(args.model_path):
    #     return
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.to(device)
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres = 2.0
    for idx, data in enumerate(dataloader):
        real_images, label, bbox, label_f, bbox_f, label_b, bbox_b = data
        real_images, label, bbox, label_f, bbox_f, label_b, bbox_b = real_images.to(device), label.long().to(device).unsqueeze(-1), bbox.float(), label_f.long().to(device), bbox_f.float(), label_b.long().to(device), bbox_b.float()
        for i in range(args.num_img):
            z_f = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().to(device)
            z_b = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().to(device)
            fake_images = netG(z_f, bbox_f, z_b, bbox_b, y_f=label_f, y_b=label_b)
            # misc.imsave("{save_path}/sample_{idx}.jpg".format(save_path=args.sample_path, idx=idx),
            #            fake_images[0].cpu().detach().numpy().transpose(1, 2, 0)*0.5+0.5)
            imageio.imwrite("{save_path}/sample_{idx}_{i}.jpg".format(save_path=args.sample_path, idx=idx, i=i),
                        fake_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--model_path', type=str, default='/home/liao/work_code/LostGANs/outputs/tmp/coco/64/model/G_200.pth', help='which epoch to load')
    parser.add_argument('--num_img', type=int, default=5, help="number of image to be generated for each layout")
    parser.add_argument('--sample_path', type=str, default='samples/tmp/64_5', help='path to save generated images')
    args = parser.parse_args()
    main(args)
