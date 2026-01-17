import os
import glob
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from slr_network import SLRModel
from utils.decode import Decode


def load_frames(frame_dir):
    img_list = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    assert len(img_list) > 0, "No frames found!"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    frames = []
    for img_path in img_list:
        img = Image.open(img_path).convert("RGB")
        img = transform(img)
        frames.append(img)

    # (T, C, H, W)
    frames = torch.stack(frames, dim=0)
    return frames


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Load gloss dict =====
    gloss_dict = np.load(args.gloss_dict, allow_pickle=True).item()
    i2g = {v[0]: k for k, v in gloss_dict.items()}

    # ===== Build model =====
    model = SLRModel(
        num_classes=args.num_classes,
        gloss_dict=gloss_dict,
        c2d_type="resnet18",
        conv_type=2,
        use_bn=1
    ).to(device)

    print(f"Loading weights from {args.weights}")
    state_dict = torch.load(args.weights, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ===== Decoder =====
    decoder = Decode(
        gloss_dict=i2g,
        num_classes=args.num_classes,
        blank_id=0,
        search_mode="max"
    )

    # ===== Load video frames =====
    frames = load_frames(args.frames_dir)
    T = frames.shape[0]

    # Add batch dim → (1, T, C, H, W)
    frames = frames.unsqueeze(0).to(device)
    vid_lgt = torch.tensor([T], dtype=torch.long).to(device)

    model.eval()

    with torch.no_grad():
        ret = model(frames, vid_lgt)

    # LẤY ĐÚNG KEY
    results = ret["recognized_sents"]   # List, batch-size length

    # batch_size = 1
    pred = results[0]                   # [(gloss, idx), (gloss, idx), ...]

    gloss_seq = [g for g, _ in pred]

    print("===== PREDICTED GLOSS =====")
    print(" ".join(gloss_seq))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", type=str, required=True,
                        help="Directory containing extracted frames (*.png)")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained model (.pt)")
    parser.add_argument("--gloss-dict", type=str, default="./VietNamese-SL/gloss_dict.npy")
    parser.add_argument("--num-classes", type=int, default=76)

    args = parser.parse_args()
    main(args)
