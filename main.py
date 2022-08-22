import os
import copy
import time
import json
import numpy as np
import torch
import datasets
import models
import argparse
from tqdm import tqdm
from losses import compute_batch_loss
import datetime
from instrumentation import train_logger


def run_train_phase(model, P, Z, logger, epoch, phase):

    '''
    Run one training phase.

    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training.
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''

    assert phase == 'train'
    model.train()

    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], phase.rjust(8, ' '))
    for batch in tqdm(Z['dataloaders'][phase], desc=desc):
        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass:
        Z['optimizer'].zero_grad()
        with torch.set_grad_enabled(True):
            batch['logits'] = model.f(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch = compute_batch_loss(batch, P, Z)
        # backward pass:
        batch['loss_tensor'].backward()
        Z['optimizer'].step()
        # save current batch data:
        logger.update_phase_data(batch)


def run_eval_phase(model, P, Z, logger, epoch, phase):

    '''
    Run one evaluation phase.

    Parameters
    model: Model to train.
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    logger: Object used to track various metrics during training.
    epoch: Integer index of the current epoch.
    phase: String giving the phase name
    '''

    assert phase in ['val', 'test']
    model.eval()
    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], phase.rjust(8, ' '))
    for batch in tqdm(Z['dataloaders'][phase], desc=desc):
        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)
        batch['labels_np'] = batch['label_vec_obs'].clone().numpy()  # copy of labels for use in metrics
        batch['label_vec_obs'] = batch['label_vec_obs'].to(Z['device'], non_blocking=True)
        # forward pass:
        with torch.set_grad_enabled(False):
            batch['logits'] = model.f(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)
            batch['preds_np'] = batch['preds'].clone().detach().cpu().numpy()  # copy of preds for use in metrics
            batch['loss_np'] = -1
            batch['reg_loss_np'] = -1
        # save current batch data:
        logger.update_phase_data(batch)


def train(model, P, Z):

    '''
    Train the model.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    Z: Dictionary of temporary objects used during training.
    '''

    best_weights_f = copy.deepcopy(model.f.state_dict())
    logger = train_logger(P) # initialize logger
    if_early_stop = False

    for epoch_idx in range(0, P['num_epochs']):
        print('start epoch [{}/{}] ...'.format(epoch_idx + 1, P['num_epochs']))
        P['epoch'] = epoch_idx + 1
        for phase in ['train', 'val', 'test']:
            # reset phase metrics:
            logger.reset_phase_data()

            # run one phase:
            t_init = time.time()
            if phase == 'train':
                run_train_phase(model, P, Z, logger, P['epoch'], phase)
                if P['epoch'] >= P['warmup_epoch'] and P['loss'] == 'EM_APL':
                    aysmmetric_pseudo_labeling(model, P, Z, logger, P['epoch'], phase)
            else:
                run_eval_phase(model, P, Z, logger, P['epoch'], phase)

            # save end-of-phase metrics:
            logger.compute_phase_metrics(phase, P['epoch'])

            # print epoch status:
            logger.report(t_init, time.time(), phase, P['epoch'])

            # update best epoch, if applicable:
            new_best = logger.update_best_results(phase, P['epoch'], P['val_set_variant'])
            if new_best:
                print('*** new best weights ***')
                best_weights_f = copy.deepcopy(model.f.state_dict())
            elif (not new_best) and (phase == 'val'):
                print('*** early stop ***')
                if_early_stop = True
                break

        if if_early_stop:
            break

    print('')
    print('*** TRAINING COMPLETE ***')
    print('Best epoch: {}'.format(logger.best_epoch))
    print('Best epoch validation score: {:.2f}'.format(logger.get_stop_metric('val', logger.best_epoch, P['val_set_variant'])))
    print('Best epoch test score:       {:.2f}'.format(logger.get_stop_metric('test', logger.best_epoch, 'clean')))

    return P, model, logger, best_weights_f


def initialize_training_run(P, feature_extractor, linear_classifier):

    '''
    Set up for model training.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''

    np.random.seed(P['seed'])

    Z = {}

    # accelerator:
    Z['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data:
    Z['datasets'] = datasets.get_data(P)

    # observed label matrix:
    label_matrix = Z['datasets']['train'].label_matrix
    num_examples = int(np.shape(label_matrix)[0])
    mtx = np.array(label_matrix).astype(np.int8)
    total_pos = np.sum(mtx == 1)
    total_neg = np.sum(mtx == 0)
    print('training samples: {} total'.format(num_examples))
    print('true positives: {} total, {:.2f} per example on average.'.format(total_pos, total_pos / num_examples))
    print('true negatives: {} total, {:.2f} per example on average.'.format(total_neg, total_neg / num_examples))
    observed_label_matrix = Z['datasets']['train'].label_matrix_obs
    num_examples = int(np.shape(observed_label_matrix)[0])
    obs_mtx = np.array(observed_label_matrix).astype(np.int8)
    obs_total_pos = np.sum(obs_mtx == 1)
    obs_total_neg = np.sum(obs_mtx == -1)
    print('observed positives: {} total, {:.2f} per example on average.'.format(obs_total_pos, obs_total_pos / num_examples))
    print('observed negatives: {} total, {:.2f} per example on average.'.format(obs_total_neg, obs_total_neg / num_examples))

    # save dataset-specific parameters:
    P['num_classes'] = Z['datasets']['train'].num_classes

    # dataloaders:
    Z['dataloaders'] = {}
    for phase in ['train', 'val', 'test']:
        Z['dataloaders'][phase] = torch.utils.data.DataLoader(
            Z['datasets'][phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False  # FIXME
        )

    # pseudo-labeling data:
    P['unlabel_num'] = []
    for i in range(observed_label_matrix.shape[1]):
        P['unlabel_num'].append(np.sum(observed_label_matrix[:, i] == 0))

    # model:
    model = models.MultilabelModel(P, feature_extractor, linear_classifier)

    # optimization objects:
    f_params = [param for param in list(model.f.parameters()) if param.requires_grad]

    Z['optimizer'] = torch.optim.Adam(
        f_params,
        lr=P['lr']
    )

    return P, Z, model


def execute_training_run(P, feature_extractor, linear_classifier):

    '''
    Initialize, run the training process, and save the results.

    Parameters
    P: Dictionary of parameters, which completely specify the training procedure.
    feature_extractor: Feature extractor model to start from.
    linear_classifier: Linear classifier model to start from.
    estimated_labels: NumPy array containing estimated training set labels to start from (for ROLE).
    '''

    P, Z, model = initialize_training_run(P, feature_extractor, linear_classifier)
    model.to(Z['device'])

    P, model, logger, best_weights_f = train(model, P, Z)

    final_logs = logger.get_logs()
    model.f.load_state_dict(best_weights_f)

    return model.f.feature_extractor, model.f.linear_classifier, final_logs


def aysmmetric_pseudo_labeling(model, P, Z, logger, epoch, phase):

    assert phase == 'train'
    model.eval()

    total_preds = None
    total_idx = None
    P['steps_per_epoch'] = len(Z['dataloaders'][phase])

    desc = '[{}/{}]{}'.format(epoch, P['num_epochs'], 'PL'.rjust(8, ' '))
    for i, batch in enumerate(tqdm(Z['dataloaders'][phase], desc=desc)):

        P['batch'] = i

        # move data to GPU:
        batch['image'] = batch['image'].to(Z['device'], non_blocking=True)

        # forward pass:
        with torch.set_grad_enabled(False):
            batch['logits'] = model.f(batch['image'])
            batch['preds'] = torch.sigmoid(batch['logits'])
            if batch['preds'].dim() == 1:
                batch['preds'] = torch.unsqueeze(batch['preds'], 0)

        # gather:
        if P['batch'] == 0:
            total_preds = batch['preds'].detach().cpu().numpy()
            total_idx = batch['idx'].cpu().numpy()
        else:
            total_preds = np.vstack((batch['preds'].detach().cpu().numpy(), total_preds))
            total_idx = np.hstack((batch['idx'].cpu().numpy(), total_idx))

            # pseudo-label:
            if P['batch'] >= P['steps_per_epoch'] - 1:

                for i in range(total_preds.shape[1]):  # class-wise

                    class_preds = total_preds[:, i]
                    class_labels_obs = Z['datasets']['train'].label_matrix_obs[:, i]
                    class_labels_obs = class_labels_obs[total_idx]

                    # select unlabel data:
                    unlabel_class_preds = class_preds[class_labels_obs == 0]
                    unlabel_class_idx = total_idx[class_labels_obs == 0]

                    # select samples:
                    neg_PL_num = int(P['neg_proportion'] * P['unlabel_num'][i] / (P['num_epochs'] - P['warmup_epoch']))
                    sorted_idx_loc = np.argsort(unlabel_class_preds)  # ascending
                    selected_idx_loc = sorted_idx_loc[:neg_PL_num]  # select indices

                    # assgin soft labels:
                    for loc in selected_idx_loc:
                        Z['datasets']['train'].label_matrix_obs[unlabel_class_idx[loc], i] = -unlabel_class_preds[loc]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='AckTheUnknown-ECCV2022')
    parser.add_argument('-g', '--gpu', default='0', choices=['0', '1', '2', '3'], type=str)
    parser.add_argument('-d', '--dataset', default='pascal', choices=['pascal', 'coco', 'nuswide', 'cub'], type=str)
    parser.add_argument('-l', '--loss', default='EM_APL', choices=['bce', 'iun', 'an', 'EM', 'EM_APL'], type=str)
    parser.add_argument('-s', '--pytorch_seed', default=0, type=int)  # try 0, 1, 8
    args = parser.parse_args()

    P = {}

    # System parameters:
    os.environ["CUDA_VISIBLE_DEVICES"] = P['GPU']
    P['pytorch_seed'] = args.pytorch_seed
    torch.manual_seed(P['pytorch_seed'])
    torch.cuda.manual_seed(P['pytorch_seed'])

    # Top-level parameters:
    P['GPU'] = args.gpu
    P['dataset'] = args.dataset
    P['loss'] = args.loss
    P['val_set_variant'] = 'clean'  # clean, observed

    # Optimization parameters:
    if P['dataset'] == 'pascal':
        P['bsize'] = 8
        P['lr'] = 1e-5
        P['alpha'] = 0.2
        P['neg_proportion'] = 0.9
        P['warmup_epoch'] = 5
        P['beta'] = 0.02
    elif P['dataset'] == 'cub':
        P['bsize'] = 8
        P['lr'] = 1e-4
        P['alpha'] = 0.01
        P['neg_proportion'] = 0.9
        P['warmup_epoch'] = 3
        P['beta'] = 0.4
    elif P['dataset'] == 'nuswide':
        P['bsize'] = 16
        P['lr'] = 1e-5
        P['alpha'] = 0.1
        P['neg_proportion'] = 0.9
        P['warmup_epoch'] = 4
        P['beta'] = 0.2
    elif P['dataset'] == 'coco':
        P['bsize'] = 16
        P['lr'] = 1e-5
        P['alpha'] = 0.1
        P['neg_proportion'] = 0.9
        P['warmup_epoch'] = 5
        P['beta'] = 0.9

    # Additional parameters:
    P['seed'] = 1200  # overall numpy seed
    P['use_pretrained'] = True  # True, False
    P['num_workers'] = 8
    P['stop_metric'] = 'map'  # metric used to select the best epoch

    # Dataset parameters:
    P['split_seed'] = 1200  # seed for train/val splitting
    P['val_frac'] = 0.2  # fraction of train set to split off for val
    P['ss_seed'] = 999  # seed for subsampling
    P['ss_frac_train'] = 1.0  # fraction of training set to subsample
    P['ss_frac_val'] = 1.0  # fraction of val set to subsample

    # Dependent parameters:
    if P['loss'] == 'bce':
        P['train_set_variant'] = 'clean'
    else:
        P['train_set_variant'] = 'observed'

    # training parameters:
    P['num_epochs'] = 10
    P['freeze_feature_extractor'] = False
    P['use_feats'] = False
    P['arch'] = 'resnet50'
    P['feature_extractor_arch'] = 'resnet50'
    P['feat_dim'] = 2048

    # run training process:
    print('[{} + {}] start exp ...'.format(P['dataset'], P['loss']))
    (feature_extractor, linear_classifier, logs) = execute_training_run(P, feature_extractor=None, linear_classifier=None)
