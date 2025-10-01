import os
import sys
import glob
import torch
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
from datetime import datetime
from utils.Functions import Dataset_epoch_cam
from model.contrast_aug import GINGroupConv
from model.car_model import VxmDense
from model.losses import NCC, l2reg_loss


def parse_args():
    parser = ArgumentParser(description="Contrast-augmented image registration training")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--checkpoint", type=int, default=20000, help="save model every N steps")
    parser.add_argument("--epochs", type=int, default=100, help="number of total epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size per gpu")
    parser.add_argument("--proj_dim", type=int, default=32, help="projection head output dimension")
    parser.add_argument("--datapath", type=str, default="../../CamCAN_2D/training", help="training data path")
    parser.add_argument("--image_size", type=tuple, default=(160, 192), help="image size")
    return parser.parse_args()


def train(args):
    torch.cuda.set_device("cuda:0")
    enc_nf = [128] * 4
    dec_nf = [256] * 7
    model = VxmDense(
        inshape=args.image_size,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=False,
        int_steps=0,
        int_downsize=2,
        out_dim=args.proj_dim,
    ).cuda()
    model.train()

    sim_loss = NCC().loss
    losses, weights = [sim_loss, l2reg_loss], [1.0, 0.3]
    contra_loss, l_contra = torch.nn.MSELoss().cuda(), 0.2
    contra_aug = GINGroupConv().cuda()

    images = sorted(glob.glob(os.path.join(args.datapath, "T1w", "sub-CC*_T1w_unbiased.nii.gz")))
    dataset = Dataset_epoch_cam(images, norm=True)
    dataloader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader), eta_min=0)

    writer = SummaryWriter("../Log/CAR_proj_32")
    model_dir = "checkpoints/CAR_proj_32"
    os.makedirs(model_dir, exist_ok=True)
    model_name = "CONTRA_AUG_NCC_0.3_0.2"

    step = 0
    for epoch in range(args.epochs):
        for X, Y in dataloader:
            X, Y = X.squeeze(-1).cuda().float(), Y.squeeze(-1).cuda().float()
            X1, X2, Y1, Y2 = contra_aug(X), contra_aug(X), contra_aug(Y), contra_aug(Y)
            y1, flow1, sv1, tv1, sd1, td1 = model(X1, X, Y1, Y)
            y2, flow2, sv2, tv2, sd2, td2 = model(X2, X, Y2, Y)

            loss = 0
            sim_losses = []
            for n, loss_fn in enumerate(losses):
                l1 = loss_fn(Y, [y1, flow1][n]) * weights[n]
                l2 = loss_fn(Y, [y2, flow2][n]) * weights[n]
                loss += l1 + l2
                sim_losses.append(l1.item())

            contra = contra_loss(sv1, sv2) + contra_loss(tv1, tv2) + contra_loss(sd1, sd2) + contra_loss(td1, td2)
            loss += l_contra * contra

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            sys.stdout.write(
                f"\rstep {step} -> sim {sim_losses[0]:.4f} | reg {sim_losses[1]:.4f} "
                f"| contra {contra.item():.4f} | total {loss.item():.4f}"
            )
            sys.stdout.flush()

            writer.add_scalar("Loss/train", loss.item(), step)
            writer.add_scalar("Loss/sim", sim_losses[0], step)
            writer.add_scalar("Loss/reg", sim_losses[1], step)
            writer.add_scalar("Loss/contra", contra.item(), step)

            # Save model
            if step % args.checkpoint == 0:
                torch.save(model.state_dict(), os.path.join(model_dir, f"{model_name}_{step}.pth"))
            step += 1
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    start = datetime.now()
    train(args)
    print("\nTraining finished in", (datetime.now() - start).total_seconds(), "seconds")
