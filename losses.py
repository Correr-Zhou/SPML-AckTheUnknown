import torch


'''
loss functions
'''


def loss_bce(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    # input validation:
    assert not torch.any(observed_labels == -1)
    assert P['train_set_variant'] == 'clean'
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None
    return loss_mtx, reg_loss


def loss_iun(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true']
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[true_labels == 0] = neg_log(1.0 - preds[true_labels == 0])  # FIXME
    reg_loss = None
    return loss_mtx, reg_loss


def loss_an(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true'].to(Z['device'])
    # input validation:
    assert torch.min(observed_labels) >= 0
    # compute loss:
    loss_mtx = torch.zeros_like(observed_labels)
    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = neg_log(1.0 - preds[observed_labels == 0])
    reg_loss = None

    return loss_mtx, reg_loss


def loss_EM(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']
    true_labels = batch['label_vec_true'].to(Z['device'])

    # input validation:
    assert torch.min(observed_labels) >= 0

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -P['alpha'] * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
        )

    return loss_mtx, None


def loss_EM_APL(batch, P, Z):
    # unpack:
    preds = batch['preds']
    observed_labels = batch['label_vec_obs']

    # input validation:
    assert torch.min(observed_labels) >= -1

    loss_mtx = torch.zeros_like(preds)

    loss_mtx[observed_labels == 1] = neg_log(preds[observed_labels == 1])
    loss_mtx[observed_labels == 0] = -P['alpha'] * (
            preds[observed_labels == 0] * neg_log(preds[observed_labels == 0]) +
            (1 - preds[observed_labels == 0]) * neg_log(1 - preds[observed_labels == 0])
        )

    soft_label = -observed_labels[observed_labels < 0]
    loss_mtx[observed_labels < 0] = P['beta'] * (
            soft_label * neg_log(preds[observed_labels < 0]) +
            (1 - soft_label) * neg_log(1 - preds[observed_labels < 0])
        )
    return loss_mtx, None



loss_functions = {
    'bce': loss_bce,
    'iun': loss_iun,
    'an': loss_an,
    'EM': loss_EM,
    'EM_APL': loss_EM_APL,
}


'''
top-level wrapper
'''


def compute_batch_loss(batch, P, Z):

    assert batch['preds'].dim() == 2

    batch_size = int(batch['preds'].size(0))
    num_classes = int(batch['preds'].size(1))

    loss_denom_mtx = (num_classes * batch_size) * torch.ones_like(batch['preds'])

    # input validation:
    assert torch.max(batch['label_vec_obs']) <= 1
    assert torch.min(batch['label_vec_obs']) >= -1
    assert batch['preds'].size() == batch['label_vec_obs'].size()
    assert P['loss'] in loss_functions

    # validate predictions:
    assert torch.max(batch['preds']) <= 1
    assert torch.min(batch['preds']) >= 0

    # compute loss for each image and class:
    loss_mtx, reg_loss = loss_functions[P['loss']](batch, P, Z)
    main_loss = (loss_mtx / loss_denom_mtx).sum()

    if reg_loss is not None:
        batch['loss_tensor'] = main_loss + reg_loss
        batch['reg_loss_np'] = reg_loss.clone().detach().cpu().numpy()
    else:
        batch['loss_tensor'] = main_loss
        batch['reg_loss_np'] = 0.0
    batch['loss_np'] = batch['loss_tensor'].clone().detach().cpu().numpy()

    return batch


'''
helper functions
'''


LOG_EPSILON = 1e-5


def neg_log(x):
    return - torch.log(x + LOG_EPSILON)