import torch
from slr_network import SLRModel
from utils.parameters import get_parser
from utils.device import set_device
from main import Processor


def load_partial_weights(model, ckpt_path):
    print(f"[INFO] Load pretrained weights from {ckpt_path}")
    state = torch.load(ckpt_path, weights_only=False)

    model_dict = model.state_dict()
    loaded, skipped = [], []

    for k, v in state.items():
        # bỏ classifier + decoder
        if (
            k.startswith("classifier")
            or k.startswith("decoder")
            or "conv1d.fc" in k
        ):
            skipped.append(k)
            continue

        if k in model_dict and model_dict[k].shape == v.shape:
            model_dict[k] = v
            loaded.append(k)
        else:
            skipped.append(k)

    model.load_state_dict(model_dict)
    print(f"[INFO] Loaded {len(loaded)} layers")
    print(f"[INFO] Skipped {len(skipped)} layers")


def main():
    parser = get_parser()
    args = parser.parse_args()

    device = set_device(args.device)

    processor = Processor(args)
    model = processor.model.to(device)

    # ĐƯỜNG DẪN CHECKPOINT PHOENIX
    phoenix_ckpt = "./work_dirs/pretrained/resnet18_slr_pretrained.pt"
    load_partial_weights(model, phoenix_ckpt)

    print("[INFO] Start finetuning on Vietnamese-SL")
    processor.start()


if __name__ == "__main__":
    main()
