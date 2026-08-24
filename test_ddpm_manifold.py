import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from gefl_f.feature_extractor import CommonFeatureExtractor
from generators.cddpm import CDDPM
import sys

def main():
    class DummyArgs:
        n_feat = 32
        n_T = 100
        guide_w = 0.3

    args = DummyArgs()
    device = 'cuda'

    # 1. Get 1000 fixed CIFAR-10 images
    transform_raw = T.Compose([T.ToTensor()])
    ds_raw = CIFAR10(root='./data', train=True, download=True, transform=transform_raw)

    idx = torch.randperm(len(ds_raw))[:1000]
    pool_imgs_raw = torch.stack([ds_raw[i][0] for i in idx]).to(device)
    pool_labels = torch.tensor([ds_raw[i][1] for i in idx]).to(device)

    # Normalize for DDPM ([-1, 1])
    norm_ddpm = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    pool_imgs_ddpm = norm_ddpm(pool_imgs_raw)

    # Normalize for Feature Extractor (standard CIFAR10)
    norm_fe = T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    pool_imgs_fe = norm_fe(pool_imgs_raw)

    print("Training feature extractor for 1 epoch on CIFAR10...", flush=True)
    # 2. Train feature extractor for 1 epoch
    fe = CommonFeatureExtractor(in_channels=3).to(device)
    opt = torch.optim.Adam(fe.parameters(), lr=1e-3)
    classifier = nn.Linear(32*16*16, 10).to(device)
    opt_cls = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    ds_train = CIFAR10(root='./data', train=True, download=True, transform=T.Compose([T.ToTensor(), norm_fe]))
    loader = DataLoader(ds_train, batch_size=128, shuffle=True)
    
    fe.train()
    for i, (x, y) in enumerate(loader):
        if i > 250: break # Just enough to get structured features
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        opt_cls.zero_grad()
        features = fe(x)
        features_flat = features.view(features.size(0), -1)
        logits = classifier(features_flat)
        loss = nn.CrossEntropyLoss()(logits, y)
        loss.backward()
        opt.step()
        opt_cls.step()

    fe.eval()
    with torch.no_grad():
        pool_features = fe(pool_imgs_fe) # right-skewed, ReLU-bounded features

    print(f"Features mean: {pool_features.mean().item():.3f}, std: {pool_features.std().item():.3f}", flush=True)
    print(f"Images mean: {pool_imgs_ddpm.mean().item():.3f}, std: {pool_imgs_ddpm.std().item():.3f}", flush=True)

    # 3. Train both DDPMs
    gen_image = CDDPM(num_classes=10, in_channels=3, img_size=32, args=args, output_activation='tanh').to(device)
    opt_img = torch.optim.Adam(gen_image.parameters(), lr=2e-4)

    # CommonFeatureExtractor outputs 32 channels, 16x16 spatial size
    gen_feat = CDDPM(num_classes=10, in_channels=32, img_size=16, args=args, output_activation='relu').to(device)
    opt_feat = torch.optim.Adam(gen_feat.parameters(), lr=2e-4)

    print(f'Step | Feat-bounded MSE | Image-Gaussian MSE')
    print(f'---- | ---------------- | ------------------')
    sys.stdout.flush()

    for step in range(3000):
        gen_image.train()
        gen_feat.train()
        
        opt_img.zero_grad()
        opt_feat.zero_grad()
        
        # Sample from the fixed pool
        batch_idx = torch.randint(0, 1000, (64,))
        batch_img = pool_imgs_ddpm[batch_idx]
        batch_feat = pool_features[batch_idx]
        batch_y = pool_labels[batch_idx]
        
        loss_img = gen_image(batch_img, batch_y)
        loss_img.backward()
        opt_img.step()
        
        loss_feat = gen_feat(batch_feat, batch_y)
        loss_feat.backward()
        opt_feat.step()
        
        if (step + 1) % 300 == 0 or step == 0:
            print(f'{step+1:4d} | {loss_feat.item():16.4f} | {loss_img.item():18.4f}', flush=True)

if __name__ == '__main__':
    main()
