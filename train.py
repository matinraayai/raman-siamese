import torch
import tqdm
from dataset import DeepCidDataset
from torch.utils.data import DataLoader
from model import get_model
import argparse


def main():
    parser = argparse.ArgumentParser("Siamese network training script.")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="Use CUDA")
    parser.add_argument("--data_path", type=str, default="data",
                        help="dataset folder path")
    parser.add_argument("--train_data_fold", type=float, default=0.7,
                        help="Amount of original data to be used in training")
    parser.add_argument("--model_path", type=str, default="model/",
                        help="path to save model checkpoints")
    parser.add_argument('--model-arch', choices=['deepcid-siamese', 'original-siamese'],
                        default='original-siamese',
                        help='architecture of the model')
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of data-loader worker threads")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Number of batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3,
                        help='Weight decay for the optimizer')
    parser.add_argument('--beta', type=tuple,
                        default=(0.5, 0.999), help="beta for the optimizer")
    parser.add_argument("--model_checkpoint_freq", type=int, default=1,
                        help="Model checkpointing frequency per number of epochs")
    parser.add_argument("--num_epochs", type=int, default=50000,
                        help="Maximum number of epochs used before stopping training")
    args = parser.parse_args()

    train_data = DeepCidDataset(args.data_path, args.train_data_fold)
    valid_data = DeepCidDataset(args.data_path, args.train_data_fold, train=False)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    loss_fn = torch.nn.BCELoss()
    net = get_model(args.model_arch)

    if args.cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr,
                                 betas=args.beta, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        train_single_epoch(net, train_loader, optimizer, loss_fn, epoch, args.cuda)
        validate_single_epoch(net, valid_loader, loss_fn, epoch, args.cuda)
        save_model(net, args.model_path + "checkpoint-%02d.cpt" % (epoch + 1), epoch, args.model_checkpoint_freq)


def train_single_epoch(net, train_loader, optimizer, loss_fn, epoch, cuda):
    net.train()
    iterator = tqdm.tqdm(train_loader)
    iterator.set_description("Epoch %d progress" % epoch)
    for batch_id, (sig1, sig2, label) in enumerate(iterator):
        if cuda:
            sig1, sig2, label = sig1.cuda(), sig2.cuda(), label.cuda()
        optimizer.zero_grad()
        output = net.forward(sig1, sig2)
        loss = loss_fn(output, label)
        iterator.set_postfix({"Loss": "%.3f" % loss.item()})
        loss.backward()
        optimizer.step()


def validate_single_epoch(net, valid_loader, loss_fn, epoch, cuda):
    net.eval()
    iterator = tqdm.tqdm(valid_loader)
    iterator.set_description("Validation for epoch %d progress" % epoch)
    true_cases = 0
    total_cases = 0
    for batch_id, (sig1, sig2, label) in enumerate(iterator):
        if cuda:
            sig1, sig2, label = sig1.cuda(), sig2.cuda(), label.cuda()
        with torch.no_grad():
            prediction = net.forward(sig1, sig2)
            loss = loss_fn(prediction, label)
            prediction.round_()
            prediction = prediction[prediction == label]
            true_cases += len(prediction)
            total_cases += len(label)
            iterator.set_postfix({"Model Loss": "%.3f" % loss.item()})
    print("Validation accuracy after epoch #%i: %.3f." % (epoch, (true_cases / total_cases)))


def save_model(net, path, epoch, frequency):
    if (epoch + 1) % frequency == 0:
        torch.save(net.state_dict(), path)
        print("Model checkpoint saved at " + path)
    return


if __name__ == '__main__':
    main()
