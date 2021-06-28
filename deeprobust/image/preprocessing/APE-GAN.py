import os
import argparse

import torch
import torch.nn 
from torch.utils.data import TensorDataset
import torch.backends.cudnn as cudnn

class Generator(nn.Module):

    def __init__(self, in_ch):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, in_ch, 4, stride=2, padding=1)

    def forward(self, x):
        h = F.leaky_relu(self.bn1(self.conv1(x)))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.deconv3(h)))
        h = F.tanh(self.deconv4(h))
        return h

class Discriminator(nn.Module):

    def __init__(self, in_ch):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2)
        self.bn3 = nn.BatchNorm2d(256)
        if in_ch == 1:
            self.fc4 = nn.Linear(1024, 1)
        else:
            self.fc4 = nn.Linear(2304, 1)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.bn2(self.conv2(h)))
        h = F.leaky_relu(self.bn3(self.conv3(h)))
        h = F.sigmoid(self.fc4(h.view(h.size(0), -1)))
        return h


def main(args):

    #Initialize GAN model
    G = Generator(in_ch = C).cuda()
    D = Discriminator(in_ch = C).cuda()

    #Initialize Generator
    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    loss_bce = nn.BCELoss()
    loss_mse = nn.MSELoss()
    cudnn.benchmark = True

    #Initialize DataLoader
    train_data = torch.load("./adv_data.tar")
    train_data = TensorDataset(train_data["normal"], train_data["adv"])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

    #Start Training
    for i in range(args.epochs):
        G.eval()
        x_fake = G(x_adv_temp).data
        G.train()
        gen_loss, dis_loss, n = 0, 0, 0
        for x, x_adv in train_loader:
            current_size = x.size(0)
            x, x_adv = x.cuda(), x_adv.cuda()

            #Train Discriminator
            t_real = torch.ones(current_size).cuda()
            t_fake = torch.zeros(current_size).cuda()
            y_real = D(x).squeeze()
            x_fake = G(x_adv)
            y_fake = D(x_fake).squeeze()

            loss_D = loss_bce(y_real, t_real) + loss_bce(y_fake, t_fake)
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train G
            for _ in range(2):
                x_fake = G(x_adv)
                y_fake = D(x_fake).squeeze()

                loss_G = args.alpha * loss_mse(x_fake, x) + args.beta * loss_bce(y_fake, t_real)
                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

            gen_loss += loss_D.data[0] * x.size(0)
            dis_loss += loss_G.data[0] * x.size(0)
            n += x.size(0)

        print("epoch:{}, LossG:{:.3f}, LossD:{:.3f}".format(i, gen_loss / n, dis_loss / n))
        torch.save({"generator": G.state_dict(), "discriminator": D.state_dict()},
                   os.path.join(args.checkpoint, "{}.tar".format(i + 1)))

    G.eval()

def get_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data", type=str, default="mnist")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="./checkpoint/test")
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    get_args()
    main(args)
