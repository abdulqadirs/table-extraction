import torch

def bce_loss(pred_labels, labels):
  pred_row, pred_col = labels

  pred_row_labels, pred_col_labels = pred_labels
  c3, c4, c5 = pred_col_labels[0], pred_col_labels[1], pred_col_labels[2]
  r3, r4, r5 = pred_row_labels[0], pred_row_labels[1], pred_row_labels[2]
  
  criterion = torch.nn.BCELoss().cuda()

  #lr3 = criterion(pred_row.view(-1), row_labels.view(-1))
  lc3 = criterion(c3.view(-1), pred_col.view(-1))
  lc4 = criterion(c4.view(-1), pred_col.view(-1))
  lc5 = criterion(c5.view(-1), pred_col.view(-1))

  lr3 = criterion(r3.view(-1), pred_row.view(-1))
  lr4 = criterion(r4.view(-1), pred_row.view(-1))
  lr5 = criterion(r5.view(-1), pred_row.view(-1))

  loss_col = lc5 + (0.25 * lc4) + (0.1 * lc3)
  loss_row = lr5 + (0.25 * lr4) + (0.1 * lr3)
  loss = loss_col + loss_row 

  return loss