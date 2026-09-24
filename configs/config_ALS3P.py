"""CLI configuration for full-information ALS3P."""

import argparse
import os


FILE_ARCH = """
./REFAVS/data
    - /media
    - /gt_mask
    - /metadata.csv
    - /audio_embed
    - /image_embed
"""

parser = argparse.ArgumentParser(
    description="SimToken Triple strategy with all-candidate soft-supervised selection"
)
parser.add_argument("--vision_pretrained", type=str, default="/home/fyl/SimToken/models/segment_anything/sam_vit_h_4b8939.pth")
parser.add_argument("--vision_tower", type=str, default="/home/fyl/huggingface/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41")
parser.add_argument("--mllm", type=str, default="/home/fyl/huggingface/models--Chat-UniVi--Chat-UniVi-7B-v1.5/snapshots/12ca2edfe2c3f9621a84c126594f66a8affce25e")

parser.add_argument("--conv_template", type=int, default=1)
parser.add_argument("--ct_weight", type=float, default=0.1)
parser.add_argument("--input_type", type=str, default="refer")
parser.add_argument("--compress", action="store_false", default=True)
parser.add_argument("--start", type=int, default=0)

parser.add_argument("--layer_strategy", choices=["selector"], default="selector")
parser.add_argument(
    "--candidate_layers",
    type=str,
    default="",
    help="Comma-separated hidden-state indices; empty uses all transformer layers.",
)
parser.add_argument("--eval_layer_strategy", choices=["argmax"], default="argmax")
parser.add_argument("--selector_loss_weight", type=float, default=0.1)
parser.add_argument("--selector_lr", type=float, default=5e-5)
parser.add_argument("--selector_warmup_epochs", type=int, default=0)
parser.add_argument("--selector_temperature_start", type=float, default=1.0)
parser.add_argument("--selector_temperature_end", type=float, default=1.0)
parser.add_argument("--selector_target_temperature", type=float, default=1.0)
parser.add_argument("--selector_max_grad_norm", type=float, default=0.5)
parser.add_argument("--map_loss_weight", type=float, default=0.0)
parser.add_argument("--map_bce_weight", type=float, default=1.0)
parser.add_argument("--map_dice_weight", type=float, default=1.0)
parser.add_argument("--gaussian_kernel", type=int, default=7)
parser.add_argument("--gaussian_sigma", type=float, default=2.0)
parser.add_argument("--grounding_grid_size", type=int, default=10)
parser.add_argument(
    "--grounding_image_size",
    type=int,
    default=224,
    help="Square padded CLIP input size for the independent Grounding view.",
)
parser.add_argument(
    "--max_sequence_length",
    type=int,
    default=4096,
    help="Fail if text, compressed tokens, and grounding tokens exceed this length.",
)
parser.add_argument("--masp_temperature", type=float, default=1.0)
parser.add_argument("--uground_lr", type=float, default=5e-5)
parser.add_argument("--uground_warmup_epochs", type=int, default=0)
parser.add_argument("--return_similarity_maps", action="store_true")

parser.add_argument("--name", type=str, default="ALS3P")
parser.add_argument(
    "--data_dir", type=str, default="data/REFAVS", help=f"Expected layout: {FILE_ARCH}"
)
parser.add_argument(
    "--saved_model",
    type=str,
    default="checkpoints/ALS3P_best_val.pth",
)
parser.add_argument("--init_checkpoint", type=str, default="")
parser.add_argument("--log_root", type=str, default="log")
parser.add_argument("--checkpoint_root", type=str, default="checkpoints")
parser.add_argument("--visualization_root", type=str, default="visualization")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--eval_batch_size", type=int, default=1)
parser.add_argument("--gpu_id", type=str, default="1")
parser.add_argument("--run", type=str, default="train")
parser.add_argument("--frame_n", type=int, default=10)
parser.add_argument("--text_max_len", type=int, default=25)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


def parse_candidate_layers(value: str):
    if not value.strip():
        return None
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(
            f"--candidate_layers must be comma-separated integers, got {value!r}"
        ) from error


args.candidate_layers = parse_candidate_layers(args.candidate_layers)

if args.layer_strategy != "selector":
    parser.error("ALS3P requires --layer_strategy selector")
if args.eval_layer_strategy != "argmax":
    parser.error("ALS3P requires --eval_layer_strategy argmax")
if args.map_loss_weight != 0.0:
    parser.error("ALS3P uses detached map quality as labels; set --map_loss_weight 0")
if args.selector_loss_weight <= 0:
    parser.error("--selector_loss_weight must be positive")
if args.selector_lr <= 0:
    parser.error("--selector_lr must be positive")
if args.selector_warmup_epochs < 0 or args.selector_warmup_epochs >= args.epochs:
    parser.error("--selector_warmup_epochs must be in [0, epochs)")
if args.selector_temperature_start <= 0 or args.selector_temperature_end <= 0:
    parser.error("selector temperatures must be positive")
if args.selector_target_temperature <= 0:
    parser.error("--selector_target_temperature must be positive")
if args.selector_max_grad_norm <= 0:
    parser.error("--selector_max_grad_norm must be positive")
