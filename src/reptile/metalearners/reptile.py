from collections import OrderedDict
from tqdm import tqdm
import numpy as np
import torch
import higher
import torch.nn.functional as F


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def mix_grad(grad_list, weight_list):
    '''
    calc weighted average of gradient
    '''
    mixed_grad = []
    for g_list in zip(*grad_list):
        g_list = torch.stack([weight_list[i] * g_list[i] for i in range(len(weight_list))])
        mixed_grad.append(torch.sum(g_list, dim=0))
    return mixed_grad


def apply_grad(model, grad):
    '''
    assign gradient to model(nn.Module) instance. return the norm of gradient
    '''
    grad_norm = 0
    for p, g in zip(model.parameters(), grad):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
        grad_norm += torch.sum(g**2)
    grad_norm = grad_norm ** (1/2)
    return grad_norm.item()


class Reptile(object):
    """Meta-learner class for Reptile [1].
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
    .. [1] Nichol, Alex, Joshua Achiam, and John Schulman. "On first-order meta-learning algorithms."
           arXiv preprint arXiv:1803.02999 (2018).
    """

    def __init__(self, model, optimizer=None, step_size=0.1, outer_step_size=0.001, first_order=False,
                 learn_step_size=False, per_param_step_size=False,
                 num_adaptation_steps=5, scheduler=None,
                 loss_function=F.cross_entropy, device=None, lr=0.001, meta_optimizer=None,
                 ohtm=False, batch_size=4, log_test_tasks=False):
        self.model = model.to(device=device)
        self.optimizer = optimizer
        self.step_size = step_size
        self.lr = lr
        self.batch_size = batch_size
        self.meta_optimizer = meta_optimizer
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_steps
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.device = device
        self.ohtm = ohtm
        self.log_test_tasks = log_test_tasks
        if self.ohtm:
            self.hardest_task = OrderedDict()
        if self.log_test_tasks:
            self.test_task_performance = OrderedDict()
        self.outer_step_size = outer_step_size
        if per_param_step_size:
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                                                             dtype=param.dtype, device=self.device,
                                                             requires_grad=learn_step_size)) for (name, param)
                                         in model.meta_named_parameters())
        else:
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                                          device=self.device, requires_grad=learn_step_size)

        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group({'params': self.step_size.values()
                                            if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                                         for group in self.optimizer.param_groups])

    @torch.enable_grad()
    def inner_loop(self, fmodel, diffopt, train_input, train_target):

        train_logit, _ = fmodel(train_input)
        inner_loss = self.loss_function(train_logit, train_target)
        diffopt.step(inner_loss)

        return None

    def outer_loop(self, batch, train=False):

        if 'test' not in batch:
            raise RuntimeError('The batch does not contain any test dataset.')
        self.model.zero_grad()
        _, test_targets, _ = batch['test']
        is_classification_task = (not test_targets.dtype.is_floating_point)
        loss_log = 0
        acc_log = 0
        loss_list = []
        grad_list = []
        self.model.zero_grad()
        for task_id, (train_input, train_target, task, test_input, test_target, _) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            train_input = train_input.to(device=self.device)
            train_target = train_target.to(device=self.device)
            test_input = test_input.to(device=self.device)
            test_target = test_target.to(device=self.device)
            with higher.innerloop_ctx(self.model, self.optimizer, track_higher_grads=True) as (fmodel, diffopt):

                for step in range(self.num_adaptation_steps):
                    if train:
                        index = np.random.permutation(np.arange(len(test_input)))[:10]
                        train_input = test_input[index]
                        train_target = test_target[index]
                    self.inner_loop(fmodel, diffopt, train_input, train_target)

                with torch.no_grad():
                    test_logit, _ = fmodel(test_input)
                    outer_loss = self.loss_function(test_logit, test_target)
                    loss_log += outer_loss.item()/self.batch_size
                    loss_list.append(outer_loss.item())
                    acc_log += get_accuracy(test_logit, test_target).item()/self.batch_size
                    if self.ohtm and train:
                        if is_classification_task:
                            self.hardest_task[str(task.cpu().tolist())
                                              ] = get_accuracy(test_logit, test_target).item()
                        else:
                            self.hardest_task[str(task.cpu().tolist())
                                              ] = -outer_loss.item()
                    if self.log_test_tasks and not train:
                        if is_classification_task:
                            self.test_task_performance[str(task.cpu().tolist())
                                                       ] = get_accuracy(test_logit, test_target).item()
                        else:
                            self.test_task_performance[str(task.cpu().tolist())
                                                       ] = outer_loss.item()

                if train:
                    outer_grad = []
                    for p_0, p_T in zip(fmodel.parameters(time=0), fmodel.parameters(time=step)):
                        outer_grad.append(-(p_T - p_0).detach())
                    grad_list.append(outer_grad)

        if train:
            weight = torch.ones(len(grad_list))/len(grad_list)
            grad = mix_grad(grad_list, weight)
            grad_log = apply_grad(self.model, grad)
            self.meta_optimizer.step()

            return loss_log, acc_log, grad_log
        else:
            return loss_log, acc_log

    def train(self, dataloader):

        loss_list = []
        acc_list = []
        grad_list = []
        with tqdm(dataloader, total=250) as pbar:
            for batch_idx, batch in enumerate(pbar):

                loss_log, acc_log, grad_log = self.outer_loop(batch, train=True)

                loss_list.append(loss_log)
                acc_list.append(acc_log)
                grad_list.append(grad_log)
                pbar.set_description('loss = {:.4f} || acc={:.4f} || grad={:.4f}'.format(
                    np.mean(loss_list), np.mean(acc_list), np.mean(grad_list)))
                if batch_idx >= 250:
                    break

        loss = np.round(np.mean(loss_list), 4)
        acc = np.round(np.mean(acc_list), 4)
        grad = np.round(np.mean(grad_list), 4)

        return loss, acc, grad

    @torch.no_grad()
    def valid(self, dataloader, max_batches=150):
        num_batches = 0
        loss_list = []
        acc_list = []
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
                loss_log, acc_log = self.outer_loop(batch, train=False)
                loss_list.append(loss_log)
                acc_list.append(acc_log)

                num_batches += 1
            if self.log_test_tasks and len(self.test_task_performance) == 1024:
                break

        loss = np.round(np.mean(loss_list), 4)
        acc = np.round(np.mean(acc_list), 4)

        return loss, acc

    def plot(self, train_inputs, train_targets, test_inputs, test_targets):
        with higher.innerloop_ctx(self.model, self.optimizer, track_higher_grads=True) as (fmodel, diffopt):
            for step in range(self.num_adaptation_steps):
                self.inner_loop(fmodel, diffopt, train_inputs, train_targets)
            with torch.no_grad():
                test_inputs = test_inputs.to(device=self.device)
                test_targets = test_targets.to(device=self.device)
                test_logits, _ = fmodel(test_inputs)

            a = test_inputs.cpu().numpy()
            b = test_targets.cpu().numpy()
            c = test_logits.cpu().numpy()
            results = [[a[i][0], b[i][0], c[i][0]] for i in range(len(a))]

        return results
