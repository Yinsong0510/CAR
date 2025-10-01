import argparse
import glob
import itertools
import os
import monai
import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from utils.Functions import dice, one_hot, Dataset_Mapping_val
from model.car_model import VxmDense
from model.layers import SpatialTransformer, calculate_jacobian_metrics

SAVEPATH = "../Log"
ENC_NF = [128] * 4
DEC_NF = [256] * 7


def evaluate_model(model, dataloader, transform_nearest, device, num_classes=4):
    """Evaluate a trained model on the validation set."""
    dice_scores, hd95_scores, grad_scores, jacobian_scores = [], [], [], []

    for X, Y, X_label, Y_label in dataloader:
        X, Y = X.squeeze(1).to(device), Y.squeeze(1).to(device)
        X_label, Y_label = X_label.squeeze(1).to(device), Y_label.squeeze(1).to(device)

        with torch.no_grad():
            _, _, F_X_Y, *_ = model(X, X, Y, Y)
            aligned_label = transform_nearest(X_label, F_X_Y)

        aligned_hot = one_hot(aligned_label, num_classes)
        fixed_hot = one_hot(Y_label, num_classes)
        hd95 = monai.metrics.compute_hausdorff_distance(aligned_hot, fixed_hot, percentile=95)[0, 1]
        hd95_scores.append(hd95.cpu().numpy())
        dice_score = dice(np.floor(aligned_label.cpu().numpy()[0, 0]), np.floor(Y_label.cpu().numpy()[0, 0]))
        dice_scores.append(dice_score)
        flow = F_X_Y.squeeze().cpu().numpy()
        fold_ratio, grad_jac = calculate_jacobian_metrics(flow)
        grad_scores.append(grad_jac)
        jacobian_scores.append(fold_ratio)

    return dice_scores, hd95_scores, grad_scores, jacobian_scores


def run(datapath, imgshape, model_dir="../Model/CAR_proj_32_mapping"):
    """Run validation on a dataset of warped pairs and labels."""
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    transform_nearest = SpatialTransformer(imgshape, mode="nearest").to(device)
    subject_dirs = sorted(glob.glob(os.path.join(datapath, "P*", "S*")))
    image_pairs, label_pairs = [], []

    for subj in subject_dirs:
        fixed = sorted(glob.glob(os.path.join(subj, "s*_t0.nii.gz")))
        moving = sorted(glob.glob(os.path.join(subj, "s*_t?_warped.nii.gz")))
        fixed_label = glob.glob(os.path.join(subj, "T1map_label.nii.gz"))
        moving_label = sorted(glob.glob(os.path.join(subj, "s*_t?_warped_label.nii.gz")))
        image_pairs += list(itertools.product(moving, fixed))
        label_pairs += list(itertools.product(moving_label, fixed_label))

    dataloader = Data.DataLoader(
        Dataset_Mapping_val(image_pairs, label_pairs, norm=True),
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    log_file = os.path.join(model_dir, "Results.txt")
    with open(log_file, "a") as log:
        log.write("Validation Results log:\n")

    model_paths = sorted(glob.glob(os.path.join(model_dir, "CONTRA_AUG_NCC_*.pth")))
    for model_path in model_paths:
        model = VxmDense(
            inshape=imgshape,
            nb_unet_features=[ENC_NF, DEC_NF],
            int_steps=0,
            int_downsize=2,
            out_dim=32,
        ).to(device)
        print("Loading weight:", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()

        dice_scores, hd95_scores, grad_scores, jacobian_scores = evaluate_model(model, dataloader, transform_nearest, device)
        with open(log_file, "a") as log:
            log.write(
                f"{model_path}:\n"
                f"Dice: {np.mean(dice_scores):.4f}, "
                f"Grad: {np.mean(grad_scores):.4f}, "
                f"HD95: {np.mean(hd95_scores):.4f}, "
                f"Fold: {np.mean(jacobian_scores):.4f}\n\n"
            )
        pd.DataFrame(dice_scores).to_csv(os.path.join(SAVEPATH, "dice_car.csv"), index=False)
        pd.DataFrame(grad_scores).to_csv(os.path.join(SAVEPATH, "grad_car.csv"), index=False)
        pd.DataFrame(jacobian_scores).to_csv(os.path.join(SAVEPATH, "jacobian_car.csv"), index=False)
        pd.DataFrame(hd95_scores).to_csv(os.path.join(SAVEPATH, "hd95_car.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for CAR mapping models")
    parser.add_argument("--datapath", type=str, required=True, help="Path to validation dataset")
    parser.add_argument("--model_dir", type=str, default="../Model/CAR_proj_32_mapping", help="Model directory")
    args = parser.parse_args()
    IMGSHAPE = (128, 128)
    run(datapath=args.datapath, imgshape=IMGSHAPE, model_dir=args.model_dir)
