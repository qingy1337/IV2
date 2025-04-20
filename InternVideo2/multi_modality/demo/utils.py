import numpy as np
import cv2
import os
import io
import copy
import torch
from torch import nn
from models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224
from models.backbones.bert.builder import build_bert
from models.criterions import get_sim
from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new
from transformers import BertTokenizer


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    if device.type == "mps":
        vid_tube = torch.from_numpy(vid_tube.astype(np.float32)).to(device, non_blocking=True).float()
    else:
        vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def get_text_feat_dict(texts, clip, text_feat_d={}):
    for t in texts:
        feat = clip.get_txt_feat(t)
        text_feat_d[t] = feat
    return text_feat_d


def get_vid_feat(frames, vlm):
    return vlm.get_vid_features(frames)

tensor_cache = {}

from collections import deque, namedtuple
PatchCacheEntry = namedtuple("PatchCacheEntry", ["frame_hash", "tokens"])

class PatchFIFO:
    """
    Holds patch tokens for the latest N frames (N = window length, 8 by default).
    Recomputes tokens only when a truly new frame arrives.
    """
    def __init__(self, model, max_frames=8):
        self.model = model          # the vision encoder
        self.max_frames = max_frames
        self.buff = deque()         # stores PatchCacheEntry
        self.device = next(model.parameters()).device

    def _to_tokens(self, frame_np):
        """
        Convert one H×W×3 BGR uint8 frame to patch tokens on self.device.
        """
        # 1)  Resize & BGR→RGB
        img = cv2.resize(frame_np[:, :, ::-1], (224, 224))
    
        # 2)  Normalise to 0‑1 and apply mean/std (same as frames2tensor)
        img = (img / 255.0 - v_mean) / v_std      # float64 nd‑array, shape [H,W,3]
    
        # 3)  → torch.float32 on correct device
        img_t = (
            torch.from_numpy(img)
            .permute(2, 0, 1)        # [3,H,W]
            .unsqueeze(0)            # [1,3,H,W]
            .unsqueeze(2)            # [1,3,T=1,H,W]
            .to(self.device, dtype=self.model.patch_embed.proj.weight.dtype)
        )
    
        # 4)  Run only the patch step (no gradients)
        with torch.no_grad():
            return self.model.patchify_frames(img_t)   # [1, 1+L, C]


    @staticmethod
    def _quick_hash(frame_np):
        # 8‑byte hash – plenty to detect exact duplicates
        return frame_np[:8,:8,:3].tobytes()

    def push(self, frame_np):
        # 0️⃣  Detect whether the model (and thus desired device) moved
        current_device = next(self.model.parameters()).device
        if current_device != self.device:
            # migrate everything we already cached
            self.buff = deque(
                PatchCacheEntry(e.frame_hash, e.tokens.to(current_device))
                for e in self.buff
            )
            self.device = current_device

        h = self._quick_hash(frame_np)
        if self.buff and self.buff[-1].frame_hash == h:
            return False  # duplicate frame, skip

        tokens = self._to_tokens(frame_np)         # already on self.device
        self.buff.append(PatchCacheEntry(h, tokens))

        if len(self.buff) > self.max_frames:
            self.buff.popleft()
        return True

    def assemble_clip(self):
        return torch.cat([e.tokens.to(self.device) for e in self.buff], dim=1)



# Updated retrieve_text with frame‑level patch embedding cache
def retrieve_text(frames,
                  texts,
                  model,
                  patch_cache,
                  topk: int = 5,
                  config: dict = {},
                  device=torch.device('cuda'),
                  log: bool = False):
    """
    Inference using frame-level patch embedding caching.
    Accepts a PatchFIFO instance to manage frame tokens.
    """
    vlm = model.to(device)
    fn = config.get('num_frames', 8)

    # 1) Push only the newest frame and wait for full window
    new_frame = frames[-1]
    patch_cache.push(new_frame)
    if len(patch_cache.buff) < fn:
        return [], np.array([])

    # 2) Assemble patch tokens and run the heavy transformer
    clip_tokens = patch_cache.assemble_clip().to(device)
    with torch.no_grad():
        pooled = vlm.vision_encoder.forward_from_patches(clip_tokens, use_image=False)
        vid_feat = vlm.vision_proj(pooled)
        vid_feat = vid_feat / vid_feat.norm(dim=-1, keepdim=True)

    # 3) Text feature retrieval (cached as before)
    calculate = any(t not in tensor_cache for t in texts)
    if calculate:
        text_feat_d = get_text_feat_dict(texts, vlm, {})
        text_feats = [text_feat_d[t] for t in texts]
        text_feats_tensor = torch.cat(text_feats, dim=0)
        for idx, t in enumerate(texts):
            tensor_cache[t] = text_feats_tensor[idx]
    else:
        if log:
            print("Using Cached text features")
        text_feats_tensor = torch.stack([tensor_cache[t] for t in texts])

    # 4) Similarity scoring
    probs, idxs = vlm.predict_label(vid_feat, text_feats_tensor, top=topk)
    ret_texts = [texts[i] for i in idxs.long().cpu().numpy()[0].tolist()]

    return ret_texts, probs.float().cpu().numpy()[0]

def setup_internvideo2(config: dict):
    if "bert" in config.model.text_encoder.name:
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
        model = InternVideo2_Stage2(config=config, tokenizer=tokenizer, is_pretrain=True)
    else:
        model = InternVideo2_Stage2(config=config, is_pretrain=True)
        tokenizer = model.tokenizer

    if config.get('compile_model', False):
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    model = model.to(torch.device(config.device))
    model_without_ddp = model

    if (config.pretrained_path.strip() and (os.path.isfile(config.pretrained_path)) or "s3://" in config.pretrained_path):
        checkpoint = torch.load(config.pretrained_path, map_location="cpu")
        try:
            if "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint["module"] # This is a deepspeed stage 1 model
        except:
            state_dict = checkpoint

        if config.get('origin_num_frames', None) is not None:
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(state_dict, model_without_ddp.vision_encoder, orig_t_size=config.origin_num_frames)
            assert a == len(state_dict), state_dict.keys()

        msg = model_without_ddp.load_state_dict(state_dict, strict=False)
        print(f"load_state_dict: {msg}")

    if config.get('use_bf16', False):
        model_without_ddp = model_without_ddp.to(torch.bfloat16)
    elif config.get('use_half_precision', False):
        model_without_ddp = model_without_ddp.to(torch.float16)
    else:
        model_without_ddp = model_without_ddp.to(torch.float32)

    model_without_ddp.eval()
    return (model_without_ddp, tokenizer,)


class InternVideo2_Stage2(nn.Module):
    """docstring for InternVideo2_Stage2"""

    def __init__(self,
                 config,
                 tokenizer,
                 is_pretrain: bool=True):
        super(InternVideo2_Stage2, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.clip_embed_dim
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.freeze_vision()

        self.text_encoder = self.build_text_encoder()
        self.freeze_text()
        self.cache_txt = {}

        self.vision_proj = nn.Linear(self.vision_width, self.embed_dim)
        self.text_proj = nn.Linear(self.text_width, self.embed_dim)

    def freeze_vision(self):
        """freeze vision encoder"""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    def encode_vision(self,
                      image: torch.Tensor,
                      test: bool=False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """

        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype) # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        # keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(
                image, None, use_image)
            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(image)
            # if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
            #     keep_temporal = False
            # print(f"\033[31mmask is {type(mask)}\033[0m")
            vision_embeds, pooled_vision_embeds, student_output, student_output_final = self.vision_encoder(
                    image, mask, use_image)
            return vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis

    def encode_text(self,
                    text: dict):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (torch.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (torch.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (torch.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (torch.Tensor): The pooled features. Shape: [B,C].

        """
        text_output = self.get_text_encoder()(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name

        if encoder_name == 'pretrain_internvideo2_1b_patch14_224':
            vision_encoder = pretrain_internvideo2_1b_patch14_224(self.config.model)
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        # parameters for mask
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_input_resolution
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (num_frames // tublet_size, img_size // patch_size, img_size // patch_size)
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio

        return vision_encoder

    def build_text_encoder(self):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = self.config.model.text_encoder.name

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def get_vid_feat(self,
                     frames: torch.Tensor):
        """get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        with torch.no_grad():
            _, vfeat = self.encode_vision(frames, test=True)
            vfeat = self.vision_proj(vfeat)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vfeat

    def get_txt_feat(self,
                     text: str):
        """get the text features for the given text."""
        if text in self.cache_txt:
            return self.cache_txt[text]
        t_original = text
        with torch.no_grad():
            text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_txt_l,
                return_tensors="pt",).to(self.config.device)
            _, tfeat = self.encode_text(text)
            tfeat = self.text_proj(tfeat)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        self.cache_txt[t_original] = tfeat
        return tfeat

    def predict_label(self,
                      vid_feat: torch.Tensor,
                      txt_feat: torch.Tensor,
                      top: int=5):
        label_probs = (100.0 * vid_feat @ txt_feat.T)
        top_probs, top_labels = label_probs.float().cpu().topk(top, dim=-1)
        return top_probs, top_labels
