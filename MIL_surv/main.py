import os
import time

from datasets.Abi_Survival import Abi_Survival
from models.AttMIL.network import MultiSurv
from models.AttMIL.engine import Engine

#from utils.options import parse_args
from utils.util import set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.util import CV_Meter
import wandb
from torch.utils.data import DataLoader, SubsetRandomSampler

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument("--excel_file", type=str, help="path to csv file")
    parser.add_argument("--folder", type=str, default="plip", help="path to features folder")

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--log_data", action="store_true", default=True, help="log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="path to latest checkpoint (default: none)")

    # Model Parameters.
    parser.add_argument("--model", type=str, default="meanmil", help="type of model (default: meanmil)")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=20, help="maximum number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="nll_surv", help="slide-level classification loss function (default: ce)")
    parser.add_argument("--alpha_surv", type=float, default=0.0, help='How much to weigh uncensored patients')
    parser.add_argument("--delta_alpha", type=float, default=0.2, help='How much to weigh delta loss')
    parser.add_argument("--num_classes", type=int, default=4, help='discretisation intervals for survival times')
    parser.add_argument("--modalities", type=str, default='clinical-wsi')
    parser.add_argument("--project_name", type=str, default='clinical-wsi')
    parser.add_argument("--study", type=str, default='os')
    args = parser.parse_args()
    return args

def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.evaluate:
        results_dir = args.resume

    else:
        results_dir = "./results/{dataset}/[{model}]-[{folder}]-[{d_a}]-[{modalities}]".format(
            dataset=args.excel_file.split("/")[-1].split(".")[0],
            model=args.model,
            folder=args.folder,
            #time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
            d_a = args.study,
            modalities = args.modalities,
        )
    print(results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # define dataset
    
    if args.folder == "PHIKON":
        args.n_features = 768
    elif args.folder == "RESNET50" or args.folder == "UNI" :
        args.n_features = 1024
    elif args.folder == "HOPTIMUS":
        args.n_features = 1536
    else:
        raise NotImplementedError("folder [{}] is not implemented".format(args.folder))
    
    dataset = Abi_Survival(excel_file=args.excel_file, folder=args.folder)

    # 5-fold cross validation
    meter = CV_Meter(fold=5)
    # start 5-fold CV evaluation.
    
    # Ensure wandb is logged in
    wandb.login()


    # Define your project and group name
    PROJECT_NAME = args.project_name 
    GROUP_NAME = "experiment-" + wandb.util.generate_id()
    
    for fold in range(5):

        ## init wandb 
        run_name = f"fold-{fold}"
        wandb.init(project=PROJECT_NAME, group=GROUP_NAME, name=run_name, job_type=run_name, config={"fold": fold})

        # get split
        train_split, val_split = dataset.get_split(fold)
        train_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(train_split))
        val_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, sampler=SubsetRandomSampler(val_split))

        # build model, criterion, optimizer, schedular
        #################################################
        # Unimodal: WSI
        data_modalities = args.modalities.split('-')
        model = MultiSurv(n_output_intervals=args.num_classes, data_modalities= data_modalities, dropout=0.25, act="relu", n_features=args.n_features,fusion_method='cat')
        engine = Engine(args, results_dir, fold)


        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, model)
        print("[model] optimizer: ", args.optimizer)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)

        # start training
        score, epoch = engine.learning(model, train_loader, val_loader, criterion, optimizer, scheduler)
        meter.updata(score, epoch)

        # End the current wandb run before the next fold starts
        wandb.finish()

    csv_path = os.path.join(results_dir, "results_{}.csv".format(args.model))
    meter.save(csv_path)


if __name__ == "__main__":
    args = parse_args()
    print(args)

    results = main(args)
    print("finished!")
