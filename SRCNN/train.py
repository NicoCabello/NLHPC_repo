import argparse
import os
import copy

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from models import SRCNN
from datasets import ImageDataset, EvalDataset
from utils import AverageMeter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-lr-file', type=str, required=True)
    parser.add_argument('--train-hr-file', type=str, required=True)
    parser.add_argument('--eval-lr-file', type=str, required=True)
    parser.add_argument('--eval-hr-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='losses')
    parser.add_argument('--output-model-dir', type=str, default='models')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max-conc', type=int, default=150)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=400)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--number-try', type=int, required=True)
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    model = SRCNN().to(device)
    print(f"The model has {sum(p.numel() for p in model.parameters())} parameters")
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr)

    train_dataset = ImageDataset(args.train_lr_file,
                                 args.train_hr_file, 
                                 args.max_conc)
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size = args.batch_size,
                                shuffle=True,
                                #   num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=True)

    eval_dataset = EvalDataset(args.eval_lr_file, 
                               args.eval_hr_file, 
                               args.max_conc)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    # check if ouput directories exists, if not, creates them
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.isdir(args.output_model_dir):
        os.mkdir(args.output_model_dir)

    # number if the try
    n_try = args.number_try

    # info of weights
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_MSE = 1000000
    n_epoch_store = 10      # save the model each 10 epochs

    train_losses = []
    val_losses = []
    train_losses_avg = []
    val_losses_avg = []

    for epoch in range(args.num_epochs):
        # print("\nStart of epoch %d" % (epoch,))
        model.train()
        epoch_losses_array = []         # Stores loss of each epoch
        epoch_losses = AverageMeter()   # Calculate loss of each epoch
        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs - 1))

            for data in train_dataloader:
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                preds = model(inputs)
                print(preds.shape)

                loss = criterion(preds, labels)
                print(loss)

                # loss of each epoch
                epoch_losses_array.append(loss.item())
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))  #corregir
                t.update(len(inputs))
            
            # storing train loss of each epoch
            train_losses.append( np.sum( epoch_losses_array ) / len(train_dataloader) )
            train_losses_avg.append(epoch_losses.avg)

        model.eval()
        val_loss = 0.0
        epoch_val_loss = AverageMeter()

        for data in eval_dataloader:
            
            inputs, labels = data
        
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs)#.clamp(-1.0, 1.0)      # only value between [-1,1]

            val_loss+=criterion(preds, labels)
            epoch_val_loss.update(criterion(preds, labels).item(), len(inputs))

        val_loss = val_loss / len(eval_dataloader)
        print('eval_MSE: {:.6f}'.format(val_loss))

        # storing validation loss of each epoch
        val_losses.append( val_loss.cpu() )
        val_losses_avg.append( epoch_val_loss.avg )
        
        if epoch % n_epoch_store == 0: # Store the model each n_epoch_store epochs
            torch.save(model.state_dict(), f'{args.output_model_dir}/model_{n_try}_epoch_{epoch}.pth')
            # np.savetxt(f"{args.output_dir}/train_losses_{n_try}.csv", train_losses, delimiter=",")
            # np.savetxt(f"{args.output_dir}/val_losses_{n_try}.csv", val_losses, delimiter=",")
            # np.savetxt(f"{args.output_dir}/train_losses_avg_{n_try}.csv", train_losses_avg, delimiter=",")
            # np.savetxt(f"{args.output_dir}/val_losses_avg_{n_try}.csv", val_losses_avg, delimiter=",")

        if val_loss < best_MSE:
            best_epoch = epoch
            best_MSE = val_loss
            best_weights = copy.deepcopy( model.state_dict() )


    print('best epoch: {}, MSE: {:.6f}'.format(best_epoch, best_MSE))
    torch.save(best_weights,f'{args.output_dir}/model_{n_try}_best_epoch_{best_epoch}.pth')

    # storing tran and validation losses
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    np.savetxt(f"{args.output_dir}/train_losses_model_{n_try}.csv", train_losses, delimiter=",")
    np.savetxt(f"{args.output_dir}/val_losses_model_{n_try}.csv", val_losses, delimiter=",")
    np.savetxt(f"{args.output_dir}/train_losses_avg_model_{n_try}.csv", train_losses_avg, delimiter=",")
    np.savetxt(f"{args.output_dir}/val_losses_avg_model_{n_try}.csv", val_losses_avg, delimiter=",")