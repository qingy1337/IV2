from configs.data import *
from configs.model import *

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
batch_size = 64 # 64 * 64
batch_size_test = 4
max_txt_l = 40

inputs = dict(
    image_res=224,
    audio_input=dict(
        audio_sample_rate=16000,
        has_multi_audio_gt=False,
        audio_reader_type='torchaudio',
        max_audio_length=10
    ),
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", audio="${max_txt_l}", video="${max_txt_l}", audio_video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", audio="${batch_size}", video="${batch_size}", audio_video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", audio="${batch_size_test}", video="${batch_size_test}", audio_video="${batch_size_test}"),
)

# ========================= model ==========================
text_enc = "bert_large"
model = dict(
    model_cls="InternVideo2_Stage2_audio",
    audio_encoder=dict(
        name='beats',
        d_model=768,
        audio_model_path="your_model_path/beats.pth",
    ),
    vision_encoder=dict(
        # backbone
        name="pretrain_internvideo2_6b_patch14_224",
        img_size=224,
        num_frames="${num_frames}",
        tubelet_size=1,
        patch_size=14,
        d_model=768,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        pretrained='your_model_path/6B_pt.pth',
        use_checkpoint=True,
        checkpoint_num=48,
        use_flash_attn=False,
        use_fused_rmsnorm=False,
        use_fused_mlp=False,
        # clip teacher
        clip_teacher=None,
        clip_input_resolution=224,
        clip_teacher_return_interval=1,
        # mask
        video_mask_type="random",
        video_mask_ratio=0.8,
        image_mask_type="random",
        image_mask_ratio=0.5,
        sep_image_video_pos_embed=False,
        keep_temporal=False,
        only_mask=True
    ),
    embed_dim=768,
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    contra_dim=768,
    av_concat_dim=768,
    temp=0.07,
    find_unused_parameters=False,
    freeze_vision=False,
    freeze_audio=True
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0,
        mlm=1.0,
        vtm=1.0,
        uta=0.0,
        # audio-related
        atc=0.0,
        avc=0.0,
        avtc=1.0,
        atm=0.0,
        avtm=1.0,
        amlm=0.0,
        avmlm=1.0
    ),  # 0: disabled.
    # ['video_name', 'selected_audio_caption', 'selected_video_caption', 'asr_captions', 'av_captions', 'video_fps', 'video_start_frame', 'video_end_frame', 'video']
    loss_caption=dict(
        # vision-related
        vtc='avs_captions',
        vtm='avs_captions',
        mlm='avs_captions',
        # audio-related
        # atc='selected_audio_caption',
        # atm='selected_audio_caption',
        # amlm='selected_audio_caption',
        # audio-vision-related
        avtc='avs_captions',
        avtm='avs_captions',
        avmlm='avs_captions',
    ),
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    distill_final_features=True,
    clip_loss_ratio=[1., 1.],
    uta_image_only=True
)

evaluate = True
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

gradient_checkpointing = True # for text encoder
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

# ========================= optimizer ==========================
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42

save_latest = False
auto_resume = True
jump_evaluate = False
pretrained_path = ""

deepspeed = dict(
    enable=True,
    stage=1,
)
