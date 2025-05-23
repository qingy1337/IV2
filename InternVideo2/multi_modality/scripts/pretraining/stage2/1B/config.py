from configs.data import *
from configs.model import *

# ========================= data ==========================
train_file = available_corpus["pretrain_example_data_1B"]

test_file = dict(msrvtt_1k_test=available_corpus["msrvtt_1k_test"],
                 didemo_ret_test=available_corpus["didemo_ret_test"])

test_types = ["msrvtt_1k_test", "didemo_ret_test"]
num_workers = 6

best_key = ["msrvtt_1k_test_match", "t2v_r1"]

# ========================= input ==========================
num_frames = 4
num_frames_test = 4
batch_size = 64 # 64 * 64
batch_size_test = 4
max_txt_l = 32

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
        random_aug=False,
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}"),
    batch_size=dict(image="${batch_size}", video="${batch_size}"),
    batch_size_test=dict(image="${batch_size_test}", video="${batch_size_test}"),
)

# ========================= model ==========================
text_enc = "bert_large"
model = dict(
    model_cls="InternVideo2_Stage2",
    vision_encoder=dict(
        # backbone
        name="pretrain_internvideo2_1b_patch14_224",
        img_size=224, 
        num_frames="${num_frames}",
        tubelet_size=1,
        patch_size=14, 
        d_model=1408,
        clip_embed_dim=768,
        clip_teacher_embed_dim=3200,
        clip_teacher_final_dim=768,
        clip_norm_type='l2',
        clip_return_layer=6,
        clip_student_return_interval=1,
        pretrained='your_model_path/1B_pt.pth',
        use_checkpoint=False,
        checkpoint_num=40,
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
        sep_image_video_pos_embed=True,
        keep_temporal=False,
        only_mask=True
    ),
    text_encoder="${TextEncoders[${text_enc}]}",
    multimodal=dict(enable=True),
    embed_dim=512,
    temp=0.07,
    find_unused_parameters=False
)

criterion = dict(
    loss_weight=dict(
        vtc=1.0, 
        mlm=1.0, 
        vtm=1.0, 
        mvm=0.0,
        uta=0.0,
    ),  # 0: disabled.
    vtm_hard_neg=True,
    mlm_masking_prob=0.5,
    distill_final_features=True,
    clip_loss_ratio=[1., 1.]
)

optimizer = dict(
    opt="adamW",
    lr=5e-5,
    opt_betas=[0.9, 0.98],  # default
    weight_decay=0.05,
    max_grad_norm=3.,  # requires a positive float, use -1 to disable
    # use a different lr for some modules, e.g., larger lr for new modules
    different_lr=dict(enable=False, module_names=[], lr=1e-3),
)

scheduler = dict(sched="cosine", epochs=10, min_lr_multi=0.01, warmup_epochs=1)

evaluate = False
deep_fusion = False
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=128,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

use_half_precision = True
use_bf16 = True

gradient_checkpointing = True # for text encoder
use_flash_sdp = False
use_mem_efficient_sdp = False and not use_flash_sdp
compile_model = False

# ========================= wandb ==========================
wandb = dict(
    enable=False,
    entity="opengvlab",  # username or team name to store the runs, see https://docs.wandb.ai/ref/python/init
    project="InternVideo2-Stage2",  # setup in your command line
)
dist_url = "env://"
device = "cuda"
mode = "pt"

# ========================= others ==========================
output_dir = None  # output dir
resume = False  # if True, load optimizer and scheduler states as well
debug = False
log_freq = 100
seed = 42

save_latest = True
auto_resume = False
jump_evaluate = False
pretrained_path = ""  # path to pretrained model weights, for resume only?
save_ckpt_iter = None
delete_ds_optim_states = True

deepspeed = dict(
    enable=True,
    stage=1,
)

