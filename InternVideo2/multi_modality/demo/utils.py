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
def retrieve_text(
    frames,
    texts,
    model,
    topk:int=5,
    config: dict={},
    device=torch.device('cuda'),
    log:bool = False
):
    vlm = model
    vlm = vlm.to(device)

    fn = config.get('num_frames', 8)
    size_t = config.get('size_t', 224)
    frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)

    if log: print(f"The frames tensor is {frames_tensor.shape} shape")

    vid_feat = vlm.get_vid_feat(frames_tensor)

    calculate = False
    for t in texts:
        if t not in tensor_cache:
            calculate = True
            break
    if calculate:
        text_feat_d = {}
        text_feat_d = get_text_feat_dict(texts, vlm, text_feat_d)
        text_feats = [text_feat_d[t] for t in texts]
        text_feats_tensor = torch.cat(text_feats, 0)
        for j in range(len(texts)):
            tensor_cache[texts[j]] = text_feats_tensor[j]
    else:
        if log: print("Using Cached")
        text_feats_tensor = torch.stack([tensor_cache[x] for x in texts])

    probs, idxs = vlm.predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.long().numpy()[0].tolist()]

    return ret_texts, probs.float().numpy()[0]

def retrieve_text_streaming(
    new_frame, # Expected: a single numpy array (HxWx3)
    texts,
    model,
    prev_hidden_state, # Expected: model-specific hidden state object (tensor or tuple)
    topk:int=5,
    config: dict={},
    device=torch.device('cuda'),
    log:bool = False
):
    """
    Performs text retrieval for a single new video frame in a streaming fashion.
    Includes added logging.
    """
    if log:
        print("\n--- retrieve_text_streaming START ---")
        print(f"Input new_frame type: {type(new_frame)}")
        if hasattr(new_frame, 'shape'):
            print(f"Input new_frame shape: {new_frame.shape}")
        print(f"Input texts type: {type(texts)}")
        print(f"Input texts list length: {len(texts)}")
        print(f"Input prev_hidden_state type: {type(prev_hidden_state)}")
        if isinstance(prev_hidden_state, torch.Tensor):
             print(f"Input prev_hidden_state shape: {prev_hidden_state.shape}")
        elif isinstance(prev_hidden_state, (list, tuple)) and len(prev_hidden_state) > 0 and isinstance(prev_hidden_state[0], torch.Tensor):
             print(f"Input prev_hidden_state (first element) shape: {prev_hidden_state[0].shape}")
        print(f"Input topk: {topk}")
        print(f"Input config: {config}")
        print(f"Input device: {device}")


    vlm = model
    vlm = vlm.to(device)

    size_t = config.get('size_t', 224)

    # New frame is a list with one frame. Pass a list containing the new_frame
    # to frames2tensor.
    frames_list_for_tensor = [new_frame]
    if log:
        print(f"Passing list of length {len(frames_list_for_tensor)} to frames2tensor.")
        if len(frames_list_for_tensor) > 0:
            print(f"Type of element being passed to frames2tensor: {type(frames_list_for_tensor[0])}")
            if hasattr(frames_list_for_tensor[0], 'shape'):
                 print(f"Shape of element being passed to frames2tensor: {frames_list_for_tensor[0].shape}")


    frames_tensor = frames2tensor(frames_list_for_tensor, fnum=1, target_size=(size_t, size_t), device=device, log=log)

    if frames_tensor is None: # Handle case where frames2tensor returned None
         if log: print("frames_tensor is None, returning empty results and prev_hidden_state.")
         return [], np.array([]), prev_hidden_state # Or None for hidden state? Depends on model


    # frames2tensor with fnum=1 returns shape [1, 1, C, H, W].
    # Squeeze out the batch (1) and time (1) dimensions resulting in [C, H, W]
    # or potentially [1, C, H, W] if the model expects a batch dimension for streaming.
    # Let's assume the model expects [1, C, H, W] for get_streaming_vid_feat input tensor.
    # So, squeeze(0) twice would result in [C, H, W], squeeze(1) once results in [1, C, H, W].
    # The original code had frames_tensor.squeeze(0) which is ambiguous if shape is [1, 1, C, H, W].
    # Let's explicitly squeeze dimension 1 to get [1, C, H, W]
    if frames_tensor.ndim == 5: # Ensure it's the expected B, T, C, H, W
         frames_tensor_input = frames_tensor.squeeze(1) # Result: [1, C, H, W]
    elif frames_tensor.ndim == 4: # Might already be [1, C, H, W] if B=1, T=1 handled internally by frames2tensor
         frames_tensor_input = frames_tensor
    else:
         if log: print(f"Unexpected frames_tensor dims {frames_tensor.ndim}: {frames_tensor.shape}")
         frames_tensor_input = frames_tensor # Proceed but potentially error

    if log: print(f"frames_tensor shape after squeeze for streaming input: {frames_tensor_input.shape}")


    # Get video features for the current frame, utilizing the previous hidden state
    if log: print("Getting streaming video features...")
    vid_feat, new_hidden_state = vlm.get_streaming_vid_feat(frames_tensor_input, prev_hidden_state = prev_hidden_state)
    if log:
        print(f"vid_feat shape: {vid_feat.shape}")
        print(f"new_hidden_state type: {type(new_hidden_state)}")
        if isinstance(new_hidden_state, torch.Tensor):
             print(f"new_hidden_state shape: {new_hidden_state.shape}")
        elif isinstance(new_hidden_state, (list, tuple)) and len(new_hidden_state) > 0 and isinstance(new_hidden_state[0], torch.Tensor):
             print(f"new_hidden_state (first element) shape: {new_hidden_state[0].shape}")


    # Check if text features need to be calculated or if cached versions can be used
    calculate = False
    for t in texts:
        if t not in tensor_cache:
            calculate = True
            break
    if log: print(f"Text feature calculation needed: {calculate}")


    if calculate:
        if log: print("Calculating text features...")
        text_feat_d = {}
        text_feat_d = get_text_feat_dict(texts, vlm, text_feat_d)
        text_feats = [text_feat_d[t] for t in texts]
        text_feats_tensor = torch.cat(text_feats, 0)
        if log: print(f"Calculated text_feats_tensor shape: {text_feats_tensor.shape}")

        # Cache calculated text features on CPU
        for j in range(len(texts)):
            tensor_cache[texts[j]] = text_feats_tensor[j].detach().cpu()
        if log: print("Cached text features.")

    else:
        if log: print("Using Cached text features.")
        # Stack cached text features into a tensor, move to device
        cached_tensors = [tensor_cache[x].to(device) for x in texts]
        text_feats_tensor = torch.stack(cached_tensors)
        if log: print(f"Stacked cached text_feats_tensor shape: {text_feats_tensor.shape}")


    # print(f"Video feature is of shape {vid_feat.shape}") # Debug print
    # print(f"Text feature is of shape {text_feats_tensor.shape}") # Debug print

    # Predict probabilities and get topk indices by comparing video and text features
    if log: print(f"Predicting labels with vid_feat shape {vid_feat.shape} and text_feats_tensor shape {text_feats_tensor.shape}")
    probs, idxs = vlm.predict_label(vid_feat, text_feats_tensor, top=topk)

    if log:
        print(f"probs shape: {probs.shape}, type: {type(probs)}")
        print(f"idxs shape: {idxs.shape}, type: {type(idxs)}")

    # Map indices back to the original text strings
    # idxs is typically [B, topk], so [0] gets the results for the first item in the batch (which is the only item)
    ret_texts = [texts[i] for i in idxs.long().numpy()[0].tolist()]

    # probs is typically [B, N] where N is number of texts. [0] gets probs for the first batch item.
    ret_probs = probs.float().numpy()[0]

    if log:
        print(f"Returning ret_texts type: {type(ret_texts)}, length: {len(ret_texts)}")
        print(f"Returning ret_probs type: {type(ret_probs)}, shape: {ret_probs.shape}")
        print(f"Returning new_hidden_state type: {type(new_hidden_state)}")
        if isinstance(new_hidden_state, torch.Tensor):
             print(f"Returning new_hidden_state shape: {new_hidden_state.shape}")
        elif isinstance(new_hidden_state, (list, tuple)) and len(new_hidden_state) > 0 and isinstance(new_hidden_state[0], torch.Tensor):
             print(f"Returning new_hidden_state (first element) shape: {new_hidden_state[0].shape}")
        print("--- retrieve_text_streaming END ---")


    # Return top texts, their probabilities, and the updated hidden state
    return ret_texts, ret_probs, new_hidden_state

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
