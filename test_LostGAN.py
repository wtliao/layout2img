import argparse
from collections import OrderedDict
import numpy as np
from scipy import misc
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.cocostuff_loader import *
from data.vg import *
from model.resnet_generator_v2 import *
from utils.util import *
import imageio
from skimage import img_as_ubyte
from tqdm import tqdm


def get_dataloader(dataset='coco', img_size=128):
    if dataset == 'coco':
        dataset = CocoSceneGraphDataset(image_dir='./datasets/coco/val2017/',
                                        instances_json='./datasets/coco/annotations/instances_val2017.json',
                                        stuff_json='./datasets/coco/annotations/stuff_val2017.json',
                                        stuff_only=True, image_size=(img_size, img_size), left_right_flip=False)
    elif dataset == 'vg':
        # with open("./datasets/vg/vocab.json", "r") as read_file:
        #     vocab = json.load(read_file)
        dataset = VgSceneGraphDataset(vocab_json='./data/tmp/vocab.json',
                                      h5_path='./data/tmp/preprocess_vg/val.h5',
                                      image_dir='./datasets/vg/',
                                      image_size=(128, 128), left_right_flip=False, max_objects=30)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1,
        drop_last=True, shuffle=False, num_workers=0)
    return dataloader


def main(args):
    device = torch.device('cuda')
    num_classes = 184 if args.dataset == 'coco' else 179
    num_o = 8 if args.dataset == 'coco' else 31
    args.num_img = 1 if args.dataset == 'vg' else 5
    
    args.sample_path = os.path.join(args.sample_path, '128_{}'.format(args.num_img))

    dataloader = get_dataloader(args.dataset)

    netG = ResnetGenerator128(num_classes=num_classes, output_dim=3).to(device)

    if not os.path.isfile(args.model_path):
        return
    state_dict = torch.load(args.model_path)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v

    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)

    netG.cuda()
    netG.eval()

    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)
    thres = 2.0

    # z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
    # z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()
    with tqdm(total=dataloader.__len__() * args.num_img) as pbar:
        for idx, data in enumerate(dataloader):
            real_images, label, bbox = data
            real_images, label, bbox = real_images.cuda(), label.long().unsqueeze(-1).cuda(), bbox.float()
            for j in range(args.num_img):
                z_obj = torch.from_numpy(truncted_random(num_o=num_o, thres=thres)).float().cuda()
                z_im = torch.from_numpy(truncted_random(num_o=1, thres=thres)).view(1, -1).float().cuda()
                fake_images = netG.forward(z_obj, bbox, z_im, label.squeeze(dim=-1))
                fake_images_uint = img_as_ubyte(fake_images[0].cpu().detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5)
                # imageio.imwrite("{save_path}/sample_{idx}_numb_{numb}.jpg".format(save_path=args.sample_path, idx=1292, numb=j), fake_images_uint)
                imageio.imwrite("{save_path}/sample_{idx}_obj_{numb}.jpg".format(save_path=args.sample_path, idx=1292, numb=idx), fake_images_uint)
                pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco',
                        help='training dataset')
    parser.add_argument('--model_path', type=str, default='/home/liao/work_code/LostGANs/samples/tmp/orig/coco/netGv2_coco128.pth', help='which epoch to load')
    #/home/liao/work_code/LostGANs/samples/tmp/orig/coco/netGv2_coco128.pth
    # /home/liao/work_code/LostGANs/outputs/tmp/orig/vg/128/model/G_200.pth
    parser.add_argument('--num_img', type=int, default=5, help="number of image to be generated for each layout")
    # parser.add_argument('--sample_path', type=str, default='samples/tmp/mask/orig/',
    #                     help='path to save generated images')
    parser.add_argument('--sample_path', type=str, default='samples/tmp/edit/orig/',
                        help='path to save generated images')
    args = parser.parse_args()
    main(args)
