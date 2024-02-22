import torch
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from src.utils import tensors_to_device
from src.protonet.metalearners.loss import prototypical_loss, get_prototypes


class PrototypicalNetwork(object):
    """Meta-learner class for Protonet [1].
    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.
    optimizer : `torch.optim.Optimizer` instance, optional
        The optimizer for the outer-loop optimization procedure. This argument
        is optional for evaluation.
    step_size : float (default: 0.1)
        The step size of the gradient descent update for fast adaptation
        (inner-loop update).
    first_order : bool (default: False)
        If `True`, then the first-order approximation of MAML is used.
    learn_step_size : bool (default: False)
        If `True`, then the step size is a learnable (meta-trained) additional
        argument [2].
    per_param_step_size : bool (default: False)
        If `True`, then the step size parameter is different for each parameter
        of the model. Has no impact unless `learn_step_size=True`.
    num_adaptation_steps : int (default: 1)
        The number of gradient descent updates on the loss function (over the
        training dataset) to be used for the fast adaptation on a new task.
    scheduler : object in `torch.optim.lr_scheduler`, optional
        Scheduler for the outer-loop optimization [3].
    loss_function : callable (default: `torch.nn.functional.cross_entropy`)
        The loss function for both the inner and outer-loop optimization.
        Usually `torch.nn.functional.cross_entropy` for a classification
        problem, of `torch.nn.functional.mse_loss` for a regression problem.
    device : `torch.device` instance, optional
        The device on which the model is defined.
    References
    ----------
    .. [1] Snell, Jake, Kevin Swersky, and Richard Zemel.
           "Prototypical networks for few-shot learning." Proceedings of the
           31st International Conference on Neural Information Processing
           Systems. 2017. (https://arxiv.org/abs/1703.05175)
    """

    def __init__(self, model, optimizer=None, scheduler=None,
                 loss_function=prototypical_loss, device=None, num_ways=None, ohtm=False, log_test_tasks=False):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.num_ways = num_ways
        self.ohtm = ohtm
        self.log_test_tasks = log_test_tasks
        if self.ohtm:
            self.hardest_task = OrderedDict()
        if self.log_test_tasks:
            self.test_task_performance = OrderedDict()

    def get_loss(self, batch, train=False):
        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')

        _, test_targets, _ = batch['test']
        num_tasks = test_targets.size(0)
        is_classification_task = (not test_targets.dtype.is_floating_point)
        results = {
            'num_tasks': num_tasks,
            'loss': np.zeros((num_tasks,), dtype=np.float32),
            'mean_loss': 0.
        }
        if is_classification_task:
            results.update({
                'accuracies': np.zeros((num_tasks,), dtype=np.float32)
            })
        self.model.zero_grad()
        mean_loss = torch.tensor(0., device=self.device)
        train_inputs, train_targets, _ = batch['train']
        test_inputs, test_targets, _ = batch['test']
        train_inputs = train_inputs.to(device=self.device)
        train_targets = train_targets.to(device=self.device)
        test_inputs = test_inputs.to(device=self.device)
        test_targets = test_targets.to(device=self.device)
        train_embeddings, _ = self.model(train_inputs)
        prototypes = get_prototypes(train_embeddings, train_targets, self.num_ways)
        test_embeddings, _ = self.model(test_inputs)
        loss, accuracy = self.loss_function(
            prototypes, test_embeddings, test_targets)
        loss.backward()

        if is_classification_task:
            results['accuracies'] = torch.mean(accuracy).item()
        if self.ohtm and train:
            for task_id, (_, _, task) in enumerate(zip(*batch['train'])):
                self.hardest_task[str(task.cpu().tolist())] = torch.mean(accuracy[task_id]).item()
        if self.log_test_tasks and not train:
            for task_id, (_, _, task) in enumerate(zip(*batch['train'])):
                self.test_task_performance[str(task.cpu().tolist())
                                           ] = torch.mean(accuracy[task_id]).item()

        mean_loss.div_(num_tasks)
        results['mean_loss'] = loss.item()

        return mean_loss, results

    def train(self, dataloader, max_batches=2000, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_loss'])}
                if 'accuracies' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        (np.mean(results['accuracies'])))
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=100):
        if self.optimizer is None:
            raise RuntimeError('Trying to call `train_iter`, while the '
                               'optimizer is `None`. In order to train `{0}`, you must '
                               'specify a Pytorch optimizer as the argument of `{0}` '
                               '(eg. `{0}(model, optimizer=torch.optim.SGD(model.'
                               'parameters(), lr=0.01), ...).'.format(__class__.__name__))
        num_batches = 0
        self.model.train()
        for batch in dataloader:
            if num_batches >= max_batches:
                break
            self.optimizer.zero_grad()

            batch = tensors_to_device(batch, device=self.device)
            loss, results = self.get_loss(batch, train=True)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            yield results
            num_batches += 1

    def evaluate(self, dataloader, max_batches=1000, verbose=True, **kwargs):
        mean_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_loss += (results['mean_loss']
                              - mean_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_loss)}
                if 'accuracies' in results:
                    mean_accuracy += (np.mean(results['accuracies'])
                                      - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_loss': mean_loss}
        if 'accuracies' in results:
            mean_results['accuracies'] = mean_accuracy

        return mean_results

    def evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.model.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if self.log_test_tasks:
                    if len(self.test_task_performance) == 1024:
                        break
                    else:
                        num_batches -= 1
                else:
                    if num_batches >= max_batches:
                        break
                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_loss(batch)
                yield results
                num_batches += 1
            if self.log_test_tasks and len(self.test_task_performance) == 1024:
                break
