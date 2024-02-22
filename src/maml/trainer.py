import json
import time
import os
import logging
import torch
from src.maml.metalearners import ModelAgnosticMetaLearning as MAML
from src.utils import get_benchmark_by_name, seed_everything
from src.datasets.task_sampler import BatchMetaDataLoader as BMD
import wandb
import numpy as np
import random


class MAMLTrainer():
    """
    MAML Trainer
    """

    def __init__(self, args):
        self.args = args
        self.highest_val = 0
        self.device = self._device()
        logging.basicConfig(level=logging.DEBUG if self.args.verbose else logging.INFO)
        logging.info(f"Configuration while training: {args}")
        self._build()

    def _build(self):
        self._create_config_file()
        self._build_loaders_and_optim()
        self._build_metalearner()
        self._train()
        wandb.save(f"{self.args.model_path}")

    def _create_config_file(self):
        if (self.args.output_folder is not None):
            if not os.path.exists(self.args.output_folder):
                os.makedirs(self.args.output_folder)
                logging.debug('Creating folder `{0}`'.format(self.args.output_folder))
            folder = os.path.join(self.args.output_folder,
                                  time.strftime('%Y-%m-%d-%H%M%S'))
            os.makedirs(folder)
            logging.debug('Creating folder `{0}`'.format(folder))

            self.args.folder = os.path.abspath(self.args.folder)
            model_dest = '{0}_model.th'.format(self.args.model)
            self.args.model_path = os.path.abspath(os.path.join(folder, model_dest))

            # Save configurations in a config.json file
            with open(os.path.join(folder, 'config.json'), 'w') as f:
                json.dump(vars(self.args), f, indent=2)
            logging.info('Saving configuration file in `{0}`'.format(
                os.path.abspath(os.path.join(folder, 'config.json'))))

    def _build_loaders_and_optim(self):
        self.benchmark = get_benchmark_by_name(self.args.model,
                                               self.args.dataset,
                                               self.args.folder,
                                               self.args.num_ways,
                                               self.args.num_shots,
                                               self.args.num_shots_test,
                                               self.args.image_size,
                                               hidden_size=self.args.hidden_size,
                                               use_augmentations=self.args.use_augmentations)
        if self.args.task_sampler == 'no_diversity_task':
            logging.info("Using no_diversity_task sampler:\n\n")
            from src.datasets.task_sampler import BatchMetaDataLoaderNDT as BMD_NDT
            self.meta_train_dataloader = BMD_NDT(self.benchmark.meta_train_dataset,
                                                 batch_size=self.args.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=True,
                                                 use_batch_collate=self.args.dataset != 'meta_dataset')
        elif self.args.task_sampler == 'no_diversity_batch':
            logging.info("Using no_diversity_batch sampler:\n\n")
            from src.datasets.task_sampler import BatchMetaDataLoaderNDB as BMD_NDB
            self.meta_train_dataloader = BMD_NDB(self.benchmark.meta_train_dataset,
                                                 batch_size=self.args.batch_size,
                                                 shuffle=True,
                                                 num_workers=self.args.num_workers,
                                                 pin_memory=True,
                                                 use_batch_collate=self.args.dataset != 'meta_dataset')
        elif self.args.task_sampler == 'no_diversity_tasks_per_batch':
            logging.info("Using no_diversity_tasks_per_batch sampler:\n\n")
            from src.datasets.task_sampler import BatchMetaDataLoaderNDTB as BMD_NDTB
            self.meta_train_dataloader = BMD_NDTB(self.benchmark.meta_train_dataset,
                                                  batch_size=self.args.batch_size,
                                                  shuffle=True,
                                                  num_workers=self.args.num_workers,
                                                  pin_memory=True,
                                                  use_batch_collate=self.args.dataset != 'meta_dataset')
        elif self.args.task_sampler == 'ohtm':
            logging.info("Using online hardest task mining sampler:\n\n")
            from src.datasets.task_sampler import OHTM
            self.meta_train_dataloader = OHTM(self.benchmark.meta_train_dataset,
                                              batch_size=self.args.batch_size,
                                              shuffle=True,
                                              num_workers=self.args.num_workers,
                                              pin_memory=True,
                                              use_batch_collate=self.args.dataset != 'meta_dataset')
        elif self.args.task_sampler == 's-DPP':
            logging.info("Using Static DPP task sampler:\n\n")
            from src.datasets.task_sampler import sDPP
            self.meta_train_dataloader = sDPP(self.benchmark.meta_train_dataset,
                                              batch_size=self.args.batch_size,
                                              shuffle=False if self.args.dataset == "meta_dataset" else True,
                                              num_workers=self.args.num_workers,
                                              pin_memory=True,
                                              dataset_name=self.args.dataset,
                                              use_batch_collate=self.args.dataset != 'meta_dataset')
        elif self.args.task_sampler == 'd-DPP':
            logging.info("Using Dynamic DPP task sampler:\n\n")
            from src.datasets.task_sampler import dDPP
            self.meta_train_dataloader = dDPP(self.benchmark.meta_train_dataset,
                                              batch_size=self.args.batch_size,
                                              shuffle=False if self.args.dataset == "meta_dataset" else True,
                                              num_workers=self.args.num_workers,
                                              pin_memory=True,
                                              model_name=self.args.model,
                                              use_batch_collate=self.args.dataset != 'meta_dataset')
        elif self.args.task_sampler == 'single_batch_fixed_pool':
            logging.info("Using single_batch_fixed_pool sampler:\n\n")
            from src.datasets.task_sampler import BatchMetaDataLoaderNDTB as BMD_NDTB
            self.meta_train_dataloader = BMD_NDTB(self.benchmark.meta_train_dataset,
                                                  batch_size=self.args.batch_size,
                                                  additional_batch_size=self.args.additional_batch_size,
                                                  shuffle=True,
                                                  num_workers=self.args.num_workers,
                                                  pin_memory=True,
                                                  use_batch_collate=self.args.dataset != 'meta_dataset')
        else:
            logging.info("Using uniform_task sampler:\n\n")
            self.meta_train_dataloader = BMD(self.benchmark.meta_train_dataset,
                                             batch_size=self.args.batch_size,
                                             shuffle=True,
                                             num_workers=self.args.num_workers,
                                             pin_memory=True,
                                             use_batch_collate=self.args.dataset != 'meta_dataset')
        self.meta_val_dataloader = BMD(self.benchmark.meta_val_dataset,
                                       batch_size=self.args.batch_size,
                                       shuffle=True,
                                       num_workers=self.args.num_workers,
                                       pin_memory=True,
                                       use_batch_collate=self.args.dataset != 'meta_dataset')

        self.meta_optimizer = torch.optim.Adam(self.benchmark.model.parameters(),
                                               lr=self.args.meta_lr)
        wandb.watch(self.benchmark.model)

    def _build_metalearner(self):

        self.metalearner = MAML(self.benchmark.model,
                                self.meta_optimizer,
                                first_order=self.args.first_order,
                                num_adaptation_steps=self.args.num_steps,
                                step_size=self.args.step_size,
                                loss_function=self.benchmark.loss_function,
                                device=self.device,
                                ohtm=self.args.task_sampler == 'ohtm')
        if self.args.task_sampler in ['ohtm', 'd-DPP']:
            self.meta_train_dataloader.init_metalearner(self.metalearner)

    def _train(self):
        best_value = None
        for epoch in range(self.args.num_epochs):
            self.metalearner.train(self.meta_train_dataloader,
                                   max_batches=self.args.num_batches,
                                   verbose=self.args.verbose,
                                   desc='Training',
                                   leave=False)
            results = self.metalearner.evaluate(self.meta_val_dataloader,
                                                max_batches=self.args.num_batches,
                                                verbose=self.args.verbose,
                                                desc='Validation')
            if (epoch+1) % self.args.log_interval == 0:
                if self.args.dataset in ["sinusoid", "sinusoid_line", "harmonic"]:
                    logging.info(f"{epoch}: {results['mean_outer_loss']}")
                    wandb.log({"Loss": results['mean_outer_loss']})
                else:
                    wandb.log({"Accuracy": results['accuracies_after']})
            # Save best model
            if self.args.dataset in ["sinusoid", "sinusoid_line", "harmonic"]:
                if ((best_value is None)
                        or (best_value > results['mean_outer_loss'])):
                    best_value = results['mean_outer_loss']
                    save_model = True
                else:
                    save_model = False
            else:
                if ((best_value is None)
                        or (best_value < results['accuracies_after'])):
                    best_value = results['accuracies_after']
                    save_model = True
                else:
                    save_model = False

            if save_model and (self.args.output_folder is not None):
                with open(self.args.model_path, 'wb') as f:
                    torch.save(self.benchmark.model.state_dict(), f)
        self.highest_val = best_value

        if hasattr(self.benchmark.meta_train_dataset, 'close'):
            self.benchmark.meta_train_dataset.close()
            self.benchmark.meta_val_dataset.close()

    def get_result(self):
        return tuple([self.highest_val])

    def _device(self):
        return torch.device('cuda' if self.args.use_cuda
                            and torch.cuda.is_available() else 'cpu')


class MAMLTester():
    """
    MAML Tester
    """

    def __init__(self, config):
        self.config = config
        self.highest_test = 0
        self.device = self._device()
        logging.basicConfig(level=logging.DEBUG if self.config['verbose'] else logging.INFO)
        logging.info(f"Configuration while testing: {config}")
        self._build()

    def _build(self):
        self._build_loader()
        self._build_metalearner()
        self._test()
        if self.config["plot"]:
            self._plot()

    def _build_loader(self):
        self.benchmark = get_benchmark_by_name(self.config['model'],
                                               self.config['dataset'],
                                               self.config['folder'],
                                               self.config['num_ways'],
                                               self.config['num_shots'],
                                               self.config['num_shots_test'],
                                               image_size=self.config['image_size'],
                                               hidden_size=self.config['hidden_size'],
                                               use_augmentations=self.config['use_augmentations'],
                                               sub_dataset_name=self.config['sub_dataset'])
        if self.config['log_test_tasks']:
            seed_everything()
            self.meta_test_dataloader = BMD(self.benchmark.meta_test_dataset,
                                            batch_size=self.config['batch_size'],
                                            shuffle=True,
                                            num_workers=0,
                                            pin_memory=True,
                                            use_batch_collate=self.config['dataset'] != 'single_meta_dataset')

        else:
            self.meta_test_dataloader = BMD(self.benchmark.meta_test_dataset,
                                            batch_size=self.config['batch_size'],
                                            shuffle=True,
                                            num_workers=self.config['num_workers'],
                                            pin_memory=True,
                                            use_batch_collate=self.config['dataset'] != 'single_meta_dataset')

        with open(self.config['model_path'], 'rb') as f:
            self.benchmark.model.load_state_dict(torch.load(f, map_location=self.device))

    def _build_metalearner(self):

        self.metalearner = MAML(self.benchmark.model,
                                first_order=self.config['first_order'],
                                num_adaptation_steps=self.config['num_steps'],
                                step_size=self.config['step_size'],
                                loss_function=self.benchmark.loss_function,
                                device=self.device,
                                log_test_tasks=self.config['log_test_tasks'])

    def _test(self):

        dirname = os.path.dirname(self.config['model_path'])
        if self.config['log_test_tasks']:
            results = self.metalearner.evaluate(self.meta_test_dataloader,
                                                max_batches=1024/self.config['batch_size'],
                                                verbose=self.config['verbose'],
                                                desc='Testing')
            self.metalearner.test_task_performance['total'] = sum(list(
                self.metalearner.test_task_performance.values()))/len(list(self.metalearner.test_task_performance.values()))
            print(f"First 10 tasks: {list(self.metalearner.test_task_performance.keys())[:10]}")
            if self.config['sub_dataset'] is not None:
                with open(os.path.join(dirname, f'task_performance_{self.config["sub_dataset"]}.json'), 'w') as f:
                    json.dump(str(self.metalearner.test_task_performance.items()), f, indent=2)
            else:
                with open(os.path.join(dirname, 'task_performance.json'), 'w') as f:
                    json.dump(str(self.metalearner.test_task_performance.items()), f, indent=2)
        else:
            results = self.metalearner.evaluate(self.meta_test_dataloader,
                                                max_batches=self.config['num_batches'],
                                                verbose=self.config['verbose'],
                                                desc='Testing')
        with open(os.path.join(dirname, 'results.json'), 'w') as f:
            json.dump(results, f)
        if self.config["dataset"] in ["sinusoid", "harmonic", "sinusoid_line"]:
            self.highest_test = results['mean_outer_loss']
        else:
            self.highest_test = results['accuracies_after']

    def get_result(self):
        return tuple([self.highest_test])

    def _device(self):
        return torch.device('cuda' if self.config['use_cuda']
                            and torch.cuda.is_available() else 'cpu')

    def _plot(self):
        results = {}
        seed_everything()
        if self.config["dataset"] == "sinusoid":
            amplitude = 1.0
            phase = 2.0
            train_inputs = np.asarray([[x] for x in random.sample(
                list([[x] for x in np.arange(-5.0, 5.0, 0.0001)]), self.config['num_shots'])])
            train_targets = amplitude * np.sin(train_inputs - phase)
            test_inputs = np.asarray([[x] for x in np.arange(-5.0, 5.0, 0.0001)])
            test_targets = amplitude * np.sin(test_inputs - phase)
            test_results = self.metalearner.plot(
                torch.tensor(train_inputs).float().to(self.device), torch.tensor(
                    train_targets).float().to(self.device),
                torch.tensor(test_inputs).float().to(self.device), torch.tensor(test_targets).float().to(self.device))
        results["train"] = [[train_inputs[i][0][0], train_targets[i][0][0]]
                            for i in range(len(train_targets))]
        results["test"] = test_results

        dirname = os.path.dirname(self.config['model_path'])
        np.save(os.path.join(dirname, 'plot_performance.npy'), results)
