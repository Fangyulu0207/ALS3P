import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import os

from torchvision import transforms
from collections import defaultdict
import cv2

# cv2.setNumThreads(0)

from PIL import Image

# from towhee import pipe, ops
from transformers import CLIPImageProcessor
from models.segment_anything.utils.transforms import ResizeLongestSide

# logger = log_agent('audio_recs.log')

import pickle as pkl
from models.llava import conversation as conversation_lib


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_GROUNDING_TOKEN = "<grounding>"

AUDIO_TOKEN_INDEX = -300
DEFAULT_AUDIO_TOKEN = "<audio>"

class REFAVSALS3P(Dataset):
    def __init__(self, split='train', cfg=None, tokenizer=None, input_type='refer'):
        self.input_type = input_type
        self.data_dir = cfg.data_dir

        meta_path = f'{self.data_dir}/metadata.csv'


        metadata = pd.read_csv(meta_path, header=0)
        self.split = split
        self.metadata = metadata[metadata['split'] == split]  # split= train,test,val.

        # 构建一个初始元素为空list的字典
        self.video_to_samples = defaultdict(list)


        for i in range(len(self.metadata)):
            row = self.metadata.iloc[i]
            vid = row['uid'].rsplit('_', 2)[0]
            self.video_to_samples[vid].append(i)


        self.all_vids = list(self.video_to_samples.keys())
        # print("all_vids", len(self.all_vids))



        self.media_path = f'{self.data_dir}/media'
        self.label_path = f'{self.data_dir}/gt_mask'
        self.frame_num = cfg.frame_n #10
        self.text_max_len = cfg.text_max_len

        self.tokenizer = tokenizer

        # 对话模板
        if cfg.conv_template == 0:
            self.system = "\nGrounding: <grounding> \nVideo: <video> \nImage: <image> \n"
        elif cfg.conv_template == 1:  #执行
            self.system = "\nGrounding: <grounding> \nVideo: <video> \nImage: <image> \nAudio: <audio> \n"


        self.question = "What is {sent} in the video?"



        self.clip_image_processor = CLIPImageProcessor.from_pretrained(cfg.vision_tower,local_files_only=True)  # vision_tower是一个冻结的CLIP图像编码器

        self.grounding_size = int(cfg.grounding_image_size)
        crop_size = self.clip_image_processor.crop_size
        if isinstance(crop_size, dict):
            clip_input_size = int(crop_size.get("height", crop_size.get("width")))
        else:
            clip_input_size = int(crop_size)
        if self.grounding_size != clip_input_size:
            raise ValueError(
                "grounding_image_size must match the loaded CLIP vision tower: "
                f"{self.grounding_size} vs {clip_input_size}"
            )
        self.grounding_transform = ResizeLongestSide(self.grounding_size)
        self.grounding_pixel_mean = torch.tensor(
            self.clip_image_processor.image_mean, dtype=torch.float32
        ).mul(255.0).view(-1, 1, 1)
        self.grounding_pixel_std = torch.tensor(
            self.clip_image_processor.image_std, dtype=torch.float32
        ).mul(255.0).view(-1, 1, 1)


        self.transform = ResizeLongestSide(1024)

        self.pixel_mean = torch.Tensor([113.263, 99.370, 92.492]).view(-1, 1, 1)
        self.pixel_std = torch.Tensor([64.274, 61.068, 58.626]).view(-1, 1, 1)
        self.img_size = 1024
        # self.img_size = 224


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def preprocess_grounding(self, image: np.ndarray):
        """UGround-style full-frame CLIP view with right/bottom padding."""
        resized = self.grounding_transform.apply_image(image)
        valid_size = tuple(int(value) for value in resized.shape[:2])
        tensor = torch.from_numpy(resized).permute(2, 0, 1).contiguous().float()
        tensor = (tensor - self.grounding_pixel_mean) / self.grounding_pixel_std
        height, width = tensor.shape[-2:]
        tensor = F.pad(
            tensor,
            (0, self.grounding_size - width, 0, self.grounding_size - height),
        )
        return tensor, valid_size



    def __len__(self):
        if self.input_type == 'refer':
            return len(self.metadata)
        elif self.input_type == 'video':
            return len(self.all_vids)

    def __getitem__(self, idx):

        if self.input_type == 'refer' :
            vid = self.metadata.iloc[idx]['uid'].rsplit('_', 2)[0]
            indices = [idx]
        elif self.input_type == 'video':

            vid = self.all_vids[idx]

            indices = self.video_to_samples[vid]


        feat_aud = torch.load(f'{self.data_dir}/audio_embed/{vid}.pt')
        image_feat = torch.load(f'{self.data_dir}/image_embed/{vid}.pt')

        img_clips = []
        img_grounding = []
        grounding_valid_sizes = []
        masks = []
        images = []
        rec_texts = []
        target_ids = []

        conversations = []

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        # print("conv.system:", conv.system)

        conv.system += self.system.format()


        for i, meta_idx in enumerate(indices):
            row = self.metadata.iloc[meta_idx]
            refer = row['exp'].lower().rstrip('.')
            fid = row['fid']

            conv.append_message(conv.roles[0], self.question.format(sent=refer))
            conv.append_message(conv.roles[1], "Sure, It is [SEG]")

            rec_texts.append(refer)
            target_ids.append(fid)

            temp_mask = []
            for frame_idx in range(self.frame_num):
                path_mask = f'{self.label_path}/{vid}/fid_{fid}/0000{frame_idx}.png'
                mask_cv2 = cv2.imread(path_mask)
                mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
                gt_binary_mask = torch.as_tensor(mask_cv2 > 0, dtype=torch.float32)
                temp_mask.append(gt_binary_mask)
            masks.append(torch.stack(temp_mask, dim=0)) # list[num_refer] :[10, 3, H, W]

        orgsize = masks[0].shape[-2:]

        conversation = conv.get_prompt()

        for _idx in range(self.frame_num):
            path_frame = f'{self.media_path}/{vid}/frames/{_idx}.jpg'
            image = cv2.imread(path_frame)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          
            image_clip = self.clip_image_processor(image, return_tensors="pt")["pixel_values"][0]  # [3, 224, 224]
            image_grounding, grounding_valid_size = self.preprocess_grounding(image)
            image = self.transform.apply_image(image)
            resize = image.shape[:2]
            image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())  # [3, 1024, 1024]

            images.append(image)
            img_clips.append(image_clip)
            img_grounding.append(image_grounding)
            grounding_valid_sizes.append(grounding_valid_size)




        images = torch.stack(images, dim=0)  # [10, 3, 1024, 1024]
        img_clips = torch.stack(img_clips, dim=0)  # [10, 3, 224, 224]
        img_grounding = torch.stack(img_grounding, dim=0)
        masks = torch.stack(masks, dim=0)  # [num_refer, 10, 3, H, W]

        return {
            'vid': vid,
            'image': images,  # [10, 3, 1024, 1024]  #原始帧，经过SAM风格预处理
            'img_clip': img_clips,  # [10, 3, 224, 224]  经过CLIP预处理的图像(resize到CLIP要求的尺寸，center crop/resize,像素归一化，转成张量pixel_values)，给MLLM提供语义用
            'img_grounding': img_grounding,
            'grounding_valid_sizes': grounding_valid_sizes,
            'mask': masks,  # [num_refer, 10, 3, H, W]
            'conversation': conversation,
            'feat_aud': feat_aud,  # [10, 128]  VGGish提取的音频特征，给MLLM提供语义用
            'resize': resize,
            'orgsize': orgsize,
            'feat_sam': image_feat,    # [T, 256, 64, 64]   SAM图像特征，为后续进行图像分割提供支持
            'ref': rec_texts,
            'fids': target_ids,
        }
