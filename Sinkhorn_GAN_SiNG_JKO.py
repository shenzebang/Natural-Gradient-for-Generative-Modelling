import argparse
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')

from geomloss import SamplesLoss
from utils import base_module_DC_GAN
# from utils import base_module_ot_gan
from torchsummary import summary
import time
import math
from tqdm import tqdm
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter


if __name__ == '__main__':

    gan_name = 'Sinkhorn_GAN'
    algorithm_name = 'SiNG'

    if not torch.cuda.is_available():
        raise NotImplementedError("SiNG only runs with CUDA.")
    device = torch.device("cuda")

    torch.set_default_tensor_type(torch.cuda.FloatTensor)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='celebA', help='dataset to be used')
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=3000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--niterG', type=int, default=3, help='no. updates of G per update of D')
    parser.add_argument('--decoder_z_features', type=int, default=64)
    parser.add_argument('--lr_encoder', type=float, default=1e-3, help='learning rate of c_encoder')
    parser.add_argument('--lr_decoder', type=float, default=1e-3, help='learning rate of μ_decoder')
    parser.add_argument('--ent_reg_loss', type=float, default=100, help='ent. reg. of the loss')
    parser.add_argument('--ent_reg_cons', type=float, default=1, help='ent. reg. of the JKO regularization')
    parser.add_argument('--jko_steps', type=int, default=20, help='# updates to solve the JKO subproblem')
    parser.add_argument('--eta', type=float, default=1, help='regularization parameter of the JKO scheme')
    parser.add_argument('--scaling', type=float, default=.95, help='scaling parameter for the Geomloss package')
    # parser.add_argument('--generator_backend', default='DC-GAN', help='NN model of the generator')
    args = parser.parse_args()

    cudnn.benchmark = True
    args.outf = "output/{}/{}/{}".format(gan_name, algorithm_name, args.dataset)
    args.modelf = "model/{}/{}/{}".format(gan_name, algorithm_name, args.dataset)
    args.dataf = "data_transform"

    if not os.path.exists(args.outf): os.makedirs(args.outf)
    if not os.path.exists(args.modelf): os.makedirs(args.modelf)


    writer = SummaryWriter(args.outf)

    time_start_0 = time.time()
    dataset = torch.load('{}/{}/{}_transform_{}.pt'.format(args.dataf, args.dataset, args.dataset, args.image_size))
    N_img, n_channels, IMAGE_SIZE_1, IMAGE_SIZE_2 = dataset.shape
    assert IMAGE_SIZE_1 == args.image_size
    assert IMAGE_SIZE_2 == args.image_size
    N_loop = int(N_img / args.batch_size)
    # Load dataset as TensorDataset
    dataset = Data.TensorDataset(dataset)
    loader = Data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    # initialize the decoder
    # if args.generator_backend == 'DC-GAN':
    μ_decoder = base_module_DC_GAN.Decoder(args.image_size, n_channels, k=args.decoder_z_features).to(device)
    # elif args.generator_backend == 'OT-GAN':
        # μ_decoder = base_module_ot_gan.Decoder(args.image_size, n_channels, k=args.decoder_z_features).to(device).to(device)
    # else:
    #     raise NotImplementedError("SiNG only implemented DC-GAN and OT-GAN.")
    μ_decoder.apply(base_module_DC_GAN.weights_init)
    optimizerμ = optim.Adam(μ_decoder.parameters(), lr=args.lr_decoder, betas=(0.5, 0.999), amsgrad=False)


    z = torch.cuda.FloatTensor(args.batch_size, args.decoder_z_features, 1, 1)
    z_observe = torch.cuda.FloatTensor(100, args.decoder_z_features, 1, 1).normal_(0, 1)
    weight = torch.ones(args.batch_size) / args.batch_size
    weight_half = torch.ones(args.batch_size // 2) / (args.batch_size / 2)



    # initialize the encoder
    c_encoder = base_module_DC_GAN.Encoder(args.image_size, n_channels).cuda()
    c_encoder.apply(base_module_DC_GAN.weights_init)
    optimizerC = optim.Adam(c_encoder.parameters(), lr=args.lr_encoder, betas=(0.5, 0.9), amsgrad=True)

    summary(c_encoder, (n_channels, args.image_size, args.image_size))
    summary(μ_decoder, (args.decoder_z_features, 1, 1))

    sinkhorn_divergence_obj = SamplesLoss(blur=math.sqrt(args.ent_reg_loss),
                                          backend="tensorized",
                                          scaling=args.scaling)
    sinkhorn_divergence_con = SamplesLoss(blur=math.sqrt(args.ent_reg_cons),
                                          backend="tensorized",
                                          scaling=args.scaling)
    i = 0
    for epoch in range(args.epochs):
        time_start = time.time()
        loss_accumulation = 0
        G_count = 0
        for data in tqdm(loader):
            x_real = data[0].to(device)
            if i % (args.niterG + 1) == 0:
                μ_detach = μ_decoder(z.normal_(0, 1)).detach()
                # --- train the encoder of ground cost:
                # ---   optimizing on the ground space endowed with cost \|φ(x) - φ(y)\|^2 is equivalent to
                # ---   optimizing on the feature space with cost \|x - y\|^2.
                optimizerC.zero_grad()
                φ_x_real_1, φ_x_real_2 = c_encoder(x_real).chunk(2, dim=0)
                φ_μ_1, φ_μ_2 = c_encoder(μ_detach).chunk(2, dim=0)
                negative_loss = - sinkhorn_divergence_obj(weight_half, φ_x_real_1, weight_half, φ_μ_1)
                negative_loss = - sinkhorn_divergence_obj(weight_half, φ_x_real_2, weight_half, φ_μ_2) + negative_loss
                negative_loss = - sinkhorn_divergence_obj(weight_half, φ_x_real_1, weight_half, φ_μ_2) + negative_loss
                negative_loss = - sinkhorn_divergence_obj(weight_half, φ_x_real_2, weight_half, φ_μ_1) + negative_loss
                negative_loss =   sinkhorn_divergence_obj(weight_half, φ_x_real_1, weight_half, φ_x_real_2) * 2 + negative_loss
                negative_loss =   sinkhorn_divergence_obj(weight_half, φ_μ_1, weight_half, φ_μ_2) * 2 + negative_loss
                # φ_x_real = c_encoder(x_real)
                # φ_μ = c_encoder(μ_decoder(z.normal_(0, 1)))
                # negative_loss = -sinkhorn_divergence(μ_weight, φ_μ, x_real_weight, φ_x_real)
                negative_loss.backward()
                optimizerC.step()

                # del φ_x_real_1, φ_x_real_2, φ_μ_1, φ_μ_2, negative_loss, μ_detach
                del negative_loss, μ_detach
                torch.cuda.empty_cache()
            else:
                G_count += 1
                # train the decoder with SiNG-JKO
                with torch.autograd.no_grad():
                    φ_μ_before = c_encoder(μ_decoder(z.normal_(0, 1)))
                    φ_x_real = c_encoder(x_real)

                # temporarily freeze the parameters of the encoder to reduce computation
                for param in c_encoder.parameters():
                    param.requires_grad = False

                for i_jko in range(args.jko_steps):
                    optimizerμ.zero_grad()
                    φ_μ = c_encoder(μ_decoder(z))
                    loss_jko = sinkhorn_divergence_obj(weight, φ_μ, weight, φ_x_real) \
                               + args.eta * sinkhorn_divergence_con(weight, φ_μ, weight, φ_μ_before)
                    loss_jko.backward()
                    optimizerμ.step()

                # unfreeze the parameters of the encoder
                for param in c_encoder.parameters():
                    param.requires_grad = True

                # evaluate the sinkhorn divergence after the JKO update
                with torch.autograd.no_grad():
                    φ_μ_after = c_encoder(μ_decoder(z))
                    S_delta = sinkhorn_divergence_con(weight, φ_μ_before, weight, φ_μ_after).item()

                # evaluate the objective value after the JKO update
                with torch.autograd.no_grad():
                    φ_μ = c_encoder(μ_decoder(z))
                    loss_print = sinkhorn_divergence_obj(weight, φ_μ, weight, φ_x_real).item()

                writer.add_scalar('GAN ent_reg_loss_{} ent_reg_cons_{} step_{} jko_steps_{} jko/loss'
                                  .format(args.ent_reg_loss, args.ent_reg_cons, args.eta, args.jko_steps), loss_print, i)
                writer.add_scalar('GAN ent_reg_loss_{} ent_reg_cons_{} step_{} jko_steps_{} jko/S_delta'
                                  .format(args.ent_reg_loss, args.ent_reg_cons, args.eta, args.jko_steps), S_delta, i)
                writer.flush()

                vutils.save_image(μ_decoder(z_observe), '{}/x_{}.png'.format(args.outf, i), normalize=True, nrow=10)

            i += 1

        torch.save(c_encoder.state_dict(), '{}/c_encoder_{}.pt'.format(args.modelf, epoch))
        torch.save(μ_decoder.state_dict(), '{}/μ_decoder_{}.pt'.format(args.modelf, epoch))
