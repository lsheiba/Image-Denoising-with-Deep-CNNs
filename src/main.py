import argparse
import torch
import torch.quantization.quantize_fx as quantize_fx
import matplotlib.pyplot as plt
from data import NoisyBSDSDataset
from argument import parse
from model import DnCNN, UDnCNN, DUDnCNN, QDUDnCNN
import nntools as nt
from utils import DenoisingStatsManager, plot


def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataset
    train_set = NoisyBSDSDataset(
        args.root_dir, image_size=args.image_size, sigma=args.sigma)
    test_set = NoisyBSDSDataset(
        args.root_dir, mode='test', image_size=args.test_image_size, sigma=args.sigma)

    # model
    if args.model == 'dncnn':
        net = DnCNN(args.D, C=args.C).to(device)
    elif args.model == 'udncnn':
        net = UDnCNN(args.D, C=args.C).to(device)
    elif args.model == 'dudncnn':
        net = DUDnCNN(args.D, C=args.C).to(device)
    elif args.model == 'quantize':
        net = QDUDnCNN(args.D, C=args.C).to(device)
    else:
        raise NameError('Please enter: dncnn, udncnn, dudncnn, or quantize')

    # optimizer
    quantize = args.quantize

    if quantize == 'fx_static':
        net.train()
        qconfig_dict = {"": torch.quantization.get_default_qat_qconfig('qnnpack')}
        net = quantize_fx.prepare_qat_fx(net, qconfig_dict)

    adam = torch.optim.Adam(net.parameters(), lr=args.lr)

    # stats manager
    stats_manager = DenoisingStatsManager()

    # experiment
    exp = nt.Experiment(
        net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
        output_dir=args.output_dir, perform_validation_during_training=True,
        device=device, quantize=args.quantize,
    )

    # run
    if args.plot:
        fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(9, 7))
        exp.run(
            num_epochs=args.num_epochs,
            plot=lambda exp: plot(exp, fig=fig, axes=axes, noisy=test_set[73][0]),
        )
    else:
        exp.run(num_epochs=args.num_epochs)


if __name__ == '__main__':
    print("MAIN")
    args = parse()
    print(args)
    run(args)
