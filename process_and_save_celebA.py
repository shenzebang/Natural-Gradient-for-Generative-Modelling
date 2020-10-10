import argparse
import os
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='tmp/celebA', help='where to save results')
    parser.add_argument('--dataf', default='data_transform/celebA', help='where to save transformed data')
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--n_particles', type=int, default=2000)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--niterD', type=int, default=3, help='no. updates of D per update of G')
    parser.add_argument('--niterG', type=int, default=1, help='no. updates of G per update of D')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.0, help='Lagrange multiplier')
    parser.add_argument('--rho', type=float, default=1e-5, help='quadratic weight penalty')
    args = parser.parse_args()

    IMAGE_SIZE = 32
    dataset = dset.CelebA(root='celeba', download=True,
                          transform=transforms.Compose([
                              transforms.Resize(IMAGE_SIZE),
                              transforms.CenterCrop(IMAGE_SIZE),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                             shuffle=True, num_workers=8, drop_last=True)

    data = next(iter(dataloader))

    os.system('mkdir -p {}'.format(args.dataf))
    torch.save(data[0], '{}/celebA_transform_{}.pt'.format(args.dataf, IMAGE_SIZE))
