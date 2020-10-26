import torch

def bce_loss(pred_labels, labels):
    pred_row, pred_col = labels
    c3, c4, c5 = pred_labels[0], pred_labels[1], pred_labels[2]
    
    criterion = torch.nn.BCELoss().cuda()

    #lr3 = criterion(pred_row.view(-1), row_labels.view(-1))
    lc3 = criterion(c3.view(-1), pred_col.view(-1))
    lc4 = criterion(c4.view(-1), pred_col.view(-1))
    lc5 = criterion(c5.view(-1), pred_col.view(-1))

    loss = lc5 + (0.25 * lc4) + (0.1 * lc3)

    return loss