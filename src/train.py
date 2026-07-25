import csv
from xml.parsers.expat import model
import nni
import time
import json
import math
import copy
import argparse
import warnings
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from models import *
from dataloader import *
from CosSimScheduler import CosSimScheduler

warnings.filterwarnings("ignore", category=Warning)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_initial_losses(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, optimizer, loss_fn, epoch):
    f1_sum = 0.0
    loss_sum = 0.0

    batch_num = math.ceil(len(index) / batch_size)
    random.shuffle(index)

    model.eval()
    with torch.no_grad():
        for batch in range(batch_num):
            if batch == batch_num - 1:
                train_idx = index[batch * batch_size:]
            else:
                train_idx = index[batch * batch_size : (batch+1) * batch_size]

            output = model(ppi_g, prot_embed, ppi_list, train_idx)
            loss = loss_fn(output, labels[train_idx])

            optimizer.zero_grad()

            loss_sum += loss.item()
            f1_score = evaluat_metrics(output.detach().cpu(), labels[train_idx].detach().cpu())
            f1_sum += f1_score


    return loss_sum / batch_num, f1_sum / batch_num



def train(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, optimizer, loss_fn, epoch):

    f1_sum = 0.0
    loss_sum = 0.0

    batch_num = math.ceil(len(index) / batch_size)
    random.shuffle(index)

    model.train()

    for batch in range(batch_num):
        if batch == batch_num - 1:
            train_idx = index[batch * batch_size:]
        else:
            train_idx = index[batch * batch_size : (batch+1) * batch_size]

        output = model(ppi_g, prot_embed, ppi_list, train_idx)
        loss = loss_fn(output, labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        f1_score = evaluat_metrics(output.detach().cpu(), labels[train_idx].detach().cpu())
        f1_sum += f1_score

        # print("Epoch: {}, Batch: {}/{} | Train Loss: {:.5f}, F1-score: {:.5f}".format(epoch, batch+1, batch_num, loss.item(), f1_score))

    return loss_sum / batch_num, f1_sum / batch_num


def evaluator(model, ppi_g, prot_embed, ppi_list, labels, index, batch_size, mode='metric'):

    eval_output_list = []
    eval_labels_list = []

    batch_num = math.ceil(len(index) / batch_size)

    model.eval()

    with torch.no_grad():
        for batch in range(batch_num):
            if batch == batch_num - 1:
                eval_idx = index[batch * batch_size:]
            else:
                eval_idx = index[batch * batch_size : (batch+1) * batch_size]

            output = model(ppi_g, prot_embed, ppi_list, eval_idx)
            eval_output_list.append(output.detach().cpu())
            eval_labels_list.append(labels[eval_idx].detach().cpu())

        f1_score = evaluat_metrics(torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0))

    if mode == 'metric':
        return f1_score
    elif mode == 'output':
        return torch.cat(eval_output_list, dim=0), torch.cat(eval_labels_list, dim=0)
    


def pretrain_vae():


    checkpoint_frequency = 10
    needed_improvement_proportion = 0.3 # If the model has not improved for 30% of the checkpoint frequency, we will return to the best checkpoint and adjust the learning rate accordingly.
    needed_improvements_epochs = needed_improvement_proportion * checkpoint_frequency
    improved_epochs = 0


    if args.pre_train is None:
        protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(param['dataset'], param['split_mode'], param['seed'])
    else:
        protein_data = load_pretrain_data(args.pre_train)

    output_dir = "../results/{}/{}/VAE/".format(param['dataset'], timestamp)
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')
    with open(os.path.join(output_dir, "config.json"), 'a+') as tf:
        json.dump(param, tf, indent=2)

    vae_dataloader = DataLoader(protein_data, batch_size=512, shuffle=True, collate_fn=collate)
    vae_model = CodeBook(param, DataLoader(protein_data, batch_size=512, shuffle=False, collate_fn=collate)).to(device)
    vae_optimizer = torch.optim.Adam(vae_model.parameters(), lr=float(param['learning_rate']), weight_decay=float(param['weight_decay']))
    if param['scheduler'] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            vae_optimizer,
            mode='min',
            factor=param['scheduler_gamma'],
            patience=param['scheduler_epochs'],
            verbose=True
        )
    elif param['scheduler'] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            vae_optimizer,
            T_max=param['pre_epoch'],
            eta_min=0,
            last_epoch=-1
        )
    elif param['scheduler'] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            vae_optimizer,
            step_size=param['scheduler_epochs'],
            gamma=param['scheduler_gamma'],
            last_epoch=-1
        )
    elif param['scheduler'] == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            vae_optimizer,
            gamma=param['scheduler_gamma'],
            last_epoch=-1
        )
    elif param['scheduler'] == "FixedLR":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            vae_optimizer,
            lr_lambda=lambda epoch: 1.0,
            last_epoch=-1
        )
    elif param['scheduler'] == "CosSim":
        scheduler = CosSimScheduler(
            vae_optimizer,
            catastrophic_gamma=param['scheduler_gamma'],
            increase_gamma=param['scheduler_epochs'],
            last_epoch=-1
        )




    best_checkpoint_loss = float("inf")
    best_checkpoint_lr = vae_optimizer.param_groups[0]["lr"]

    torch.save({
        "model": copy.deepcopy(vae_model.state_dict()),
        "optimizer": copy.deepcopy(vae_optimizer.state_dict()),
    }, "best_vae_state.pth")


    epoch_loss = 0
    num_batches = 0
    for iter_num, batch_graph in enumerate(vae_dataloader):
        batch_graph.to(device)

        z, e, e_q_loss, recon_loss, mask_loss = vae_model(batch_graph)
        loss_vae = e_q_loss + recon_loss + mask_loss * param['mask_loss']


        if (epoch - 1) % param['log_num'] == 0 and iter_num == 0:
            print("\033[0;30;43m Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\033[0m".format(epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
            log_file.write("Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\n".format(epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
            log_file.flush()

        epoch_loss += loss_vae.item()
        num_batches += 1

    epoch_loss /= num_batches
    vae_optimizer.zero_grad()
    
    train_losses = [epoch_loss]


    


    for epoch in range(1, param["pre_epoch"] + 1):
        epoch_loss = 0.0
        num_batches = 0



        for iter_num, batch_graph in enumerate(vae_dataloader):

            batch_graph.to(device)

            z, e, e_q_loss, recon_loss, mask_loss = vae_model(batch_graph)
            loss_vae = e_q_loss + recon_loss + mask_loss * param['mask_loss']

            vae_optimizer.zero_grad()
            loss_vae.backward()
            vae_optimizer.step()

            if (epoch - 1) % param['log_num'] == 0 and iter_num == 0:
                print("\033[0;30;43m Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\033[0m".format(epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.write("Pre-training VQ-VAE | Epoch: {}, Batch: {} | Train Loss: {:.5f} | {:.5f} {:.5f} {:.5f}\n".format(epoch, iter_num, loss_vae.item(), e_q_loss.item(), recon_loss.item(), mask_loss.item()))
                log_file.flush()

            epoch_loss += loss_vae.item()
            num_batches += 1

        epoch_loss /= num_batches

        train_losses.append(epoch_loss)

        print(f"Current vs. Previous loss: {epoch_loss:.3e}:{train_losses[-2]:.3e}")

        loss_ratio = epoch_loss / train_losses[-2]

        if(loss_ratio < 1):
            improved_epochs += 1

        if param['scheduler'] == "ReduceLROnPlateau":
            scheduler.step(epoch_loss)

        elif param['scheduler'] == "CosSim":

            if loss_ratio > 2:
                print("Loss exploded, rolling back.")

                ckpt = torch.load("best_vae_state.pth")
                vae_model.load_state_dict(ckpt["model"])
                vae_optimizer.load_state_dict(ckpt["optimizer"])

            scheduler.cos_step(
                cos_sim=-1,
                loss_ratio=loss_ratio,
                old_lr=None
            )

        else:
            scheduler.step()

        if epoch % checkpoint_frequency == 0:
            if epoch_loss < best_checkpoint_loss:

                torch.save({
                    "model": copy.deepcopy(vae_model.state_dict()),
                    "optimizer": copy.deepcopy(vae_optimizer.state_dict()),
                }, "best_vae_state.pth")

                best_checkpoint_loss = epoch_loss
                best_checkpoint_lr = vae_optimizer.param_groups[0]["lr"]

            elif (
                improved_epochs < needed_improvements_epochs
                and isinstance(scheduler, CosSimScheduler)
            ):

                print("Rolling back VAE checkpoint")


                ckpt = torch.load("best_vae_state.pth")
                vae_model.load_state_dict(ckpt["model"])
                vae_optimizer.load_state_dict(ckpt["optimizer"])


                scheduler.cos_step(
                    cos_sim=-1,
                    loss_ratio=loss_ratio,
                    old_lr=best_checkpoint_lr
                )

            improved_epochs = 0            
    ckpt = torch.load("best_vae_state.pth")

    vae_model.load_state_dict(ckpt["model"])
    torch.save(vae_model.state_dict(), os.path.join(output_dir, f'vae_model.ckpt'))

    del vae_model
    torch.cuda.empty_cache()



def main():

    protein_data, ppi_g, ppi_list, labels, ppi_split_dict = load_data(param['dataset'], param['split_mode'], param['seed'])
    
    vae_model = CodeBook(param, DataLoader(protein_data, batch_size=512, shuffle=False, collate_fn=collate)).to(device)
    if args.ckpt_path is None:
        vae_model.load_state_dict(torch.load(os.path.join("../results/{}/{}/VAE/".format(param['dataset'], timestamp), f'vae_model.ckpt')))
    else:
        vae_model.load_state_dict(torch.load(args.ckpt_path))
    prot_embed = vae_model.Protein_Encoder.forward(vae_model.vq_layer).to(device)

    del vae_model
    torch.cuda.empty_cache()

    output_dir = "../results/{}/{}/SEES_{}/".format(param['dataset'], timestamp, param['seed'])
    check_writable(output_dir, overwrite=False)
    log_file = open(os.path.join(output_dir, "train_log.txt"), 'a+')

    model = GIN(param).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(param['learning_rate']), weight_decay=float(param['weight_decay']))
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    print(param.keys())

    if param['scheduler'] == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=param['scheduler_gamma'], patience=param['scheduler_epochs'], verbose=True)
    elif param['scheduler'] == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=param['max_epochs'], eta_min=0, last_epoch=-1)
    elif param['scheduler'] == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param['scheduler_epochs'], gamma=param['scheduler_gamma'], last_epoch=-1)
    elif param['scheduler'] == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=param['scheduler_gamma'], last_epoch=-1)
    elif param['scheduler'] == "FixedLR":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0, last_epoch=-1)
    elif param['scheduler'] == "CosSim":
            scheduler = CosSimScheduler(optimizer, catastrophic_gamma=param['scheduler_gamma'], increase_gamma=param['scheduler_epochs'], last_epoch=-1)

    es = 0
    val_best = 0
    test_val = 0
    test_best = 0
    best_epoch = 0
    checkpoint_frequency = 20
    needed_improvement_proportion = 0.3 # If the model has not improved for 30% of the checkpoint frequency, we will return to the best checkpoint and adjust the learning rate accordingly.
    needed_improvements_epochs = needed_improvement_proportion * checkpoint_frequency
    improved_epochs = 0

    best_model_path = "best_model_state.pth"

    best_checkpoint_loss, best_train_f1 = get_initial_losses(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['train_index'], param['batch_size'], optimizer, loss_fn, 0)

    best_checkpoint_lr = optimizer.param_groups[0]['lr']



    torch.save({
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }, best_model_path)


    train_losses = [best_checkpoint_loss] # large initial values so that the first


    for epoch in range(1, param["max_epoch"] + 1):
        
        train_loss, train_f1_score = train(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['train_index'], param['batch_size'], optimizer, loss_fn, epoch)
        
        train_losses.append(train_loss)

        loss_ratio = train_losses[-1] / best_checkpoint_loss

        if(train_losses[-1] < train_losses[-2]):
            improved_epochs += 1




        if(param['scheduler'] == "ReduceLROnPlateau"):
            scheduler.step(train_loss)
        elif(param['scheduler'] == "CosSim"):
            if(loss_ratio > 2 and isinstance(scheduler, CosSimScheduler)):
                print("Loss is more than 2x best loss, rolling back and reducing learning rate by catastrophic gamma: {}".format(scheduler.catastrophic_gamma))

                ckpt = torch.load(best_model_path)
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])

            scheduler.cos_step(cos_sim=-1, loss_ratio=loss_ratio, old_lr=None)
        else:
            scheduler.step()


        if (epoch - 1) % param['log_num'] == 0:

            val_f1_score = evaluator(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['val_index'], param['batch_size'])
            test_f1_score = evaluator(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['test_index'], param['batch_size'])

            if test_f1_score > test_best:
                test_best = test_f1_score

            if val_f1_score >= val_best:
                val_best = val_f1_score
                test_val = test_f1_score
                es = 0
                best_epoch = epoch
            else:
                es += 1

            if(param['scheduler'] == "ReduceLROnPlateau"):
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = scheduler.get_last_lr()[0]

            print("Best Learning Rate: {:.4e}".format(best_checkpoint_lr))
            print(f"Number of improved epochs vs. required: {improved_epochs} : {needed_improvements_epochs}")
            print("\033[0;30;46m Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Learning Rate: {:.4e} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\033[0m".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, current_lr, val_best, test_val, test_best, best_epoch))
            log_file.write(" Epoch: {}, Train Loss: {:.5f} | Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Learning Rate: {:.4e} | Val Best: {:.4f}, Test Val: {:.4f}, Test Best: {:.4f} | Best Epoch: {}\n".format(
                    epoch, train_loss, train_f1_score, val_f1_score, test_f1_score, current_lr, val_best, test_val, test_best, best_epoch))
            log_file.flush()

            if es == 500:
                print("Early stopping!")
                break


        if epoch % checkpoint_frequency == 0:
            print("Checkpointing model at epoch {}".format(epoch))
            if(train_loss < best_checkpoint_loss):

                torch.save({
                    "model": copy.deepcopy(model.state_dict()),
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                }, best_model_path)
                best_checkpoint_loss = train_loss
                best_train_f1 = train_f1_score
                best_checkpoint_lr = optimizer.param_groups[0]['lr']
            elif(improved_epochs < needed_improvements_epochs and isinstance(scheduler, CosSimScheduler)):
                print(f"Model has not improved for {needed_improvement_proportion * 100}% of epochs, rolling back to best checkpoint and reducing learning rate by catastrophic gamma: {scheduler.catastrophic_gamma}")
                ckpt = torch.load(best_model_path)
                model.load_state_dict(ckpt["model"])
                optimizer.load_state_dict(ckpt["optimizer"])

                scheduler.cos_step(cos_sim=-1, loss_ratio=loss_ratio, old_lr=best_checkpoint_lr)


            improved_epochs = 0



    log_file.close()
    ckpt = torch.load(best_model_path)
    model.load_state_dict(ckpt["model"])
    eval_output, eval_labels = evaluator(model, ppi_g, prot_embed, ppi_list, labels, ppi_split_dict['test_index'], param['batch_size'], 'output')

    np.save(os.path.join(output_dir, "eval_output.npy"), eval_output.detach().cpu().numpy())
    np.save(os.path.join(output_dir, "eval_labels.npy"), eval_labels.detach().cpu().numpy())

    # jsobj = json.dumps(ppi_split_dict)
    # with open(os.path.join(output_dir, "ppi_split_dict.json"), 'w') as f:
    #     f.write(jsobj)
    #     f.close()

    return test_f1_score, test_val, test_best, best_epoch



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DGL implementation")
    parser.add_argument("--dataset", type=str, default="SHS27k")
    parser.add_argument("--split_mode", type=str, default="random")
    parser.add_argument("--input_dim", type=int, default=7)
    parser.add_argument("--output_dim", type=int, default=7)
    parser.add_argument("--ppi_hidden_dim", type=int, default=512)
    parser.add_argument("--prot_hidden_dim", type=int, default=128)
    parser.add_argument("--ppi_num_layers", type=int, default=2)
    parser.add_argument("--prot_num_layers", type=int, default=4)
    
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--max_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--dropout_ratio", type=float, default=0.0)
    
    parser.add_argument("--pre_epoch", type=int, default=50)
    parser.add_argument("--commitment_cost", type=float, default=0.25)
    parser.add_argument("--num_embeddings", type=int, default=512)
    parser.add_argument("--mask_ratio", type=float, default=0.15)
    parser.add_argument("--sce_scale", type=float, default=1.5)
    parser.add_argument("--mask_loss", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_num", type=int, default=1)
    parser.add_argument("--data_mode", type=int, default=0)
    parser.add_argument("--data_split_mode", type=int, default=0)
    parser.add_argument("--pre_train", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)

    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau", choices=["ReduceLROnPlateau", "CosineAnnealingLR", "StepLR", "ExponentialLR", "FixedLR", "CosSim"])
    # gamma for expontialLR + stepLR, factor for reduce on plateau, what we decrease by in cossim
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    # patience for reduce on plateau, step_size for stepLR, increase amount for cossim
    parser.add_argument("--scheduler_epochs", type=float, default=10)

    args = parser.parse_args()

    # Start with CLI arguments
    param = vars(args).copy()

    # Load defaults from config
    if os.path.exists("../configs/param_configs.json"):
        config = json.loads(
            open("../configs/param_configs.json").read()
        )[param["dataset"]][param["split_mode"]]

        # JSON provides defaults, CLI overrides
        config.update(param)
        param = config

    # Finally let NNI override everything
    param.update(nni.get_next_parameter())
    timestamp = time.strftime("%Y-%m-%d %H-%M-%S") + f"-%3d" % ((time.time() - int(time.time())) * 1000)
    print("Starting training at {}".format(timestamp))


    if param['data_mode'] == 0:
        param['dataset'] = 'SHS27k'
    elif param['data_mode'] == 1:
        param['dataset'] = 'SHS148k'
    elif param['data_mode'] == 2:
        param['dataset'] = 'STRING'

    if param['data_split_mode'] == 0:
        param['split_mode'] = 'random'
    elif param['data_split_mode'] == 1:
        param['split_mode'] = 'bfs'
    elif param['data_split_mode'] == 2:
        param['split_mode'] = 'dfs'

    set_seed(param['seed'])

    # breakpoint()
    if args.ckpt_path is None:
        pretrain_vae()
    test_acc, test_val, test_best, best_epoch = main()
    nni.report_final_result(test_val)

    outFile = open('../PerformMetrics_Metrics.csv','a+', newline='')
    writer = csv.writer(outFile, dialect='excel')
    results = [timestamp]
    for v, k in param.items():
        results.append(k)
    
    results.append(str(test_acc))
    results.append(str(test_val))
    results.append(str(test_best))
    writer.writerow(results)
