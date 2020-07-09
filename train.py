import torch
import pickle
from dataset import RamanDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from model import Siamese1D
import time
import numpy as np
from collections import deque
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Siamese network training script.")
    parser.add_argument("--cuda", type=bool, default=True,
                        help="Use CUDA")
    parser.add_argument("--data_path", type=str, default="data",
                        help="dataset folder path")
    parser.add_argument("--train_data_fold", type=float, default=0.7,
                        help="Amount of original data to be used in training")
    parser.add_argument("--model_path", type=str, default="model/siamese",
                        help="path to save model checkpoints")
    parser.add_argument("--n_way", type=int, default=40,
                        help="number of ways for one-shot learning")
    parser.add_argument("--accuracy_test_samples", type=int, default=40,
                        help="number of samples to test accuracy")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of data-loader worker threads")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of batch size")
    parser.add_argument("--lr", type=float, default=0.00006,
                        help="learning rate")
    # gflags.DEFINE_integer("show_every", 10, "show result after each show_every iter.")
    parser.add_argument("--model_checkpoint_freq", type=int, default=100,
                        help="Model checkpointing frequency per number of iterations")
    parser.add_argument("--model_test_freq", type=int, default=100,
                        help="Model testing frequency per number of iterations")
    parser.add_argument("--max_iter", type=int, default=50000,
                        help="Maximum number of iterations used before stopping training")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    print("use gpu:", args.gpu_ids, "to train.")

    train_data = RamanDataset(args.data_path, args.train_data_fold)
    test_data = RamanDataset(args.data_path, args.train_data_fold, train=False)

    trainLoader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testLoader = DataLoader(test_data, batch_size=args.n_way, shuffle=False, num_workers=args.num_workers)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    net = Siamese1D(0.2)

    # multi gpu
    if len(args.gpu_ids.split(",")) > 1:
        net = torch.nn.DataParallel(net)

    if args.cuda:
        net.cuda()

    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    optimizer.zero_grad()

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (img1, img2, label) in enumerate(trainLoader, 1):
        if batch_id > args.max_iter:
            break
        if args.cuda:
            img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
        else:
            img1, img2, label = Variable(img1), Variable(img2), Variable(label)
        optimizer.zero_grad()
        output = net.forward(img1, img2)
        loss = loss_fn(output, label)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % args.show_every == 0 :
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s'%(batch_id, loss_val/Flags.show_every, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % args.save_every == 0:
            torch.save(net.state_dict(), Flags.model_path + '/model-inter-' + str(batch_id+1) + ".pt")
        if batch_id % Flags.test_every == 0:
            right, error = 0, 0
            for _, (test1, test2) in enumerate(testLoader, 1):
                if args.cuda:
                    test1, test2 = test1.cuda(), test2.cuda()
                output = net.forward(test1, test2).data.cpu().numpy()
                pred = np.argmax(output)
                if pred == 0:
                    right += 1
                else:
                    error += 1
            print('*'*70)
            print('[%d]\tTest set\tcorrect:\t%d\terror:\t%d\tprecision:\t%f'%(batch_id, right, error, right*1.0/(right+error)))
            print('*'*70)
            queue.append(right * 1.0 / (right + error))
        train_loss.append(loss_val)

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#"*70)
    print("final accuracy: ", acc/20)
