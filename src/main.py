import tensorflow as tf
import argparse
from src.logger import Logger
import gc
import torch
import json
from glob import glob
from src.utils import seed_everything
from src.maml import MAMLTrainer, MAMLTester
from src.protonet import ProtonetTrainer, ProtonetTester
from src.reptile import ReptileTrainer, ReptileTester
from src.matching_networks import MatchingNetworksTrainer, MatchingNetworksTester
from src.cnaps import CNAPTrainer, CNAPTester
from src.metaoptnet import MetaOptNetTrainer, MetaOptNetTester
import wandb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet the TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet the TensorFlow warnings


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser('Task_Diversity')
    # General
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of experimental runs (default: 5).')
    parser.add_argument('--model', type=str,
                        choices=['maml', 'protonet', 'reptile',
                                 'matching_networks', 'cnaps', 'metaoptnet'],
                        default='maml',
                        help='Name of the model to be used (default: MAML).')
    parser.add_argument('--task_sampler', type=int,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        default=0,
                        help='Type of task sampler to be used '
                        '(default: 0).')
    parser.add_argument('--folder', type=str,
                        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--dataset', type=str,
                        choices=['sinusoid', 'omniglot', 'miniimagenet',
                                 'tiered_imagenet', 'meta_dataset', 'harmonic', 'sinusoid_line'],
                        default='omniglot',
                        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--sub-dataset', type=str,
                        default='ilsvrc_2012',
                        help='Name of the dataset (default: omniglot).')
    parser.add_argument('--output-folder', type=str, default=None,
                        help='Path to the output folder to save the model.')
    parser.add_argument('--num-ways', type=int, default=5,
                        help='Number of classes per task (N in "N-way",'
                        ' default: 5).')
    parser.add_argument('--num-shots', type=int, default=5,
                        help='Number of training example per class '
                        '(k in "k-shot", default: 5).')
    parser.add_argument('--num-shots-test', type=int, default=15,
                        help='Number of test example per class.'
                        ' If negative, same as the number '
                        'of training examples `--num-shots-test` (default: 15).')

    # Model
    parser.add_argument('--hidden-size', type=int, default=64,
                        help='Number of channels in each convolution '
                        'layer of the VGG network (default: 64).')
    parser.add_argument('--image-size', type=int, default=84,
                        help='Image size (default: 84).')

    # Optimization
    parser.add_argument('--batch-size', type=int, default=25,
                        help='Number of tasks in a batch of tasks '
                        '(default: 25).')
    parser.add_argument('--additional-batch-size', default=None,
                        help='Number of repetitions of given task in single batch fixed pool sampler'
                        '(default: None).')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='Number of fast adaptation steps, '
                        'ie. gradient descent updates (default: 1).')
    parser.add_argument('--num-epochs', type=int, default=50,
                        help='Number of epochs of meta-training '
                        '(default: 50).')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='Number of batch of tasks per epoch '
                        '(default: 100).')
    parser.add_argument('--step-size', type=float, default=0.1,
                        help='Size of the fast adaptation step,'
                        ' ie. learning rate in the '
                        'gradient descent update (default: 0.1).')
    parser.add_argument('--first-order', action='store_true',
                        help='Use the first order approximation,'
                        ' do not use higher-order derivatives during '
                        'meta-optimization.')
    parser.add_argument('--meta-lr', type=float, default=0.001,
                        help='Learning rate for the meta-optimizer '
                        '(optimization of the outer loss). The default '
                        'optimizer is Adam (default: 0.001).')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer '
                        '(optimization of the outer loss). The default '
                        'optimizer is Adam (default: 1e-3).')
    parser.add_argument('--lr_scheduler_step', type=int, default=20,
                        help='StepLR learning rate scheduler step, (default=20).')
    parser.add_argument('--lr_scheduler_gamma', type=float, default=0.5,
                        help='Learning rate for the StepLR scheduler.'
                        '(default: 0.5).')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer.'
                        '(default: 0.0001).')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum for optimizer.'
                        '(default: 0.5).')

    # Misc
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of workers to use for data-loading '
                        '(default: 1).')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--use-augmentations', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--log-interval', type=int, default=1,
                        help='Log interval of the model '
                        '(default: 1 epoch).')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name'
                        '(default: None).')
    parser.add_argument('--log-test-tasks', action='store_true')
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    if args.num_shots_test <= 0:
        args.num_shots_test = args.num_shots
    return args


def get_task_sampler(choice):
    task_sampler = {0: 'uniform',
                    1: 'no_diversity_task',
                    2: 'no_diversity_episode',
                    3: 'no_diversity_tasks_per_episode',
                    4: 'ohtm',
                    5: 'owhtm',
                    6: 's-DPP',
                    7: 'd-DPP',
                    8: 'single_batch_fixed_pool',
                    9: 'single_batch_increased_data',
                    10: 'adaptive_sampler'}
    return task_sampler[choice]


def test_model(args, dataset_name=None):
    log = Logger(len(glob(f'{args.output_folder}/*/config.json')))
    log.reset(len(glob(f'{args.output_folder}/*/config.json')), info="Testing Accuracy")
    for run, config_file in enumerate(glob(f'{args.output_folder}/*/config.json')):
        with open(config_file, 'r') as f:
            config = json.load(f)
        if args.folder is not None:
            config['folder'] = args.folder
        if args.num_steps > 0:
            config['num_steps'] = args.num_steps
        if args.num_batches > 0:
            config['num_batches'] = args.num_batches
        if args.exp_name is not None:
            config['exp_name'] = args.exp_name
        for arg in vars(args):
            if arg not in config.keys():
                config[arg] = getattr(args, arg)
        config['verbose'] = args.verbose
        config['train'] = args.train
        config['dataset'] = args.dataset
        config['log_test_tasks'] = args.log_test_tasks
        config['plot'] = args.plot
        config['sub_dataset'] = args.sub_dataset
        config['batch_size'] = args.batch_size
        if not args.train and (dataset_name is None or dataset_name == 'ilsvrc_2012'):
            wandb.init(project='Task_Diversity', entity='td_ml', config=config, settings=wandb.Settings(start_method='thread'),
                       name=config['exp_name'], reinit=False)
            wandb.config = config
        if config['model'] == 'maml':
            """
            MAML Test
            """
            maml_tester = MAMLTester(config)
            log.add_result(run, maml_tester.get_result())
        elif config['model'] == 'protonet':
            """
            Protonet Test
            """
            protonet_tester = ProtonetTester(config)
            log.add_result(run, protonet_tester.get_result())
        elif config['model'] == 'reptile':
            """
            Reptile Test
            """
            reptile_tester = ReptileTester(config)
            log.add_result(run, reptile_tester.get_result())
        elif config['model'] == 'matching_networks':
            """
            MatchingNetworks Test
            """
            mn_tester = MatchingNetworksTester(config)
            log.add_result(run, mn_tester.get_result())
        elif config['model'] == 'cnaps':
            """
            Conditional Neural Adaptive Processes Test
            """
            cnap_tester = CNAPTester(config)
            log.add_result(run, cnap_tester.get_result())
        elif config['model'] == 'metaoptnet':
            """
            MetaOptNet Test
            """
            mon_tester = MetaOptNetTester(config)
            log.add_result(run, mon_tester.get_result())
    if dataset_name is not None:
        print(f"Average Performance of {config['model']} on {args.dataset}/{dataset_name}:")
    else:
        print(f"Average Performance of {config['model']} on {args.dataset}:")
    log.print_statistics()


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.empty_cache()
    args.task_sampler = get_task_sampler(args.task_sampler)
    if args.task_sampler == 'uniform':
        args.batch_size = 1
    if args.task_sampler == 'single_batch_fixed_pool':
        args.additional_batch_size = args.batch_size
        args.num_epochs = args.num_epochs*args.batch_size
        args.batch_size = 1
    if args.task_sampler == 'single_batch_increased_data':
        args.num_epochs = args.num_epochs*args.batch_size
        args.batch_size = 1
    args.exp_name = f"{args.exp_name}_{args.task_sampler}_sampler"
    if args.train:
        wandb.init(project='Task_Diversity', entity='td_ml', config=args, name=args.exp_name,
                   settings=wandb.Settings(start_method='thread'), reinit=False)
        log = Logger(args.runs)
        log.reset(args.runs, info="Validation Accuracy")

        if args.model == 'maml':
            """
            MAML Trainer
            """
            for run in range(args.runs):
                gc.collect()
                seed_everything(run)
                maml_trainer = MAMLTrainer(args)
                log.add_result(run, maml_trainer.get_result())
        elif args.model == 'protonet':
            """
            Protonet Trainer
            """
            for run in range(args.runs):
                gc.collect()
                seed_everything(run)
                protonet_trainer = ProtonetTrainer(args)
                log.add_result(run, protonet_trainer.get_result())
        elif args.model == 'reptile':
            """
            Reptile Trainer
            """
            for run in range(args.runs):
                gc.collect()
                seed_everything(run)
                reptile_trainer = ReptileTrainer(args)
                log.add_result(run, reptile_trainer.get_result())
        elif args.model == 'matching_networks':
            """
            MatchingNetworks Trainer
            """
            for run in range(args.runs):
                gc.collect()
                seed_everything(run)
                mn_trainer = MatchingNetworksTrainer(args)
                log.add_result(run, mn_trainer.get_result())
        elif args.model == 'cnaps':
            """
            Conditional Neural Adaptive Processes Trainer
            """
            for run in range(args.runs):
                gc.collect()
                seed_everything(run)
                cnaps_trainer = CNAPTrainer(args)
                log.add_result(run, cnaps_trainer.get_result())
        elif args.model == 'metaoptnet':
            """
            MetaOptNet Trainer
            """
            for run in range(args.runs):
                gc.collect()
                seed_everything(run)
                mon_trainer = MetaOptNetTrainer(args)
                log.add_result(run, mon_trainer.get_result())

        print(f"Average Performance of {args.model} on {args.dataset}:")
        log.print_statistics()
    args.sub_dataset = None
    if args.dataset == 'meta_dataset':
        if not args.train:
            # for dataset in ["ilsvrc_2012", "omniglot", "aircraft", "cu_birds", "dtd", "quickdraw", "fungi",
            #                 "vgg_flower", "traffic_sign", "mscoco"]:
            for dataset in ["dtd"]:
                args.dataset = 'single_meta_dataset'
                args.sub_dataset = dataset
                test_model(args, dataset)
    else:
        test_model(args)
    wandb.finish()
