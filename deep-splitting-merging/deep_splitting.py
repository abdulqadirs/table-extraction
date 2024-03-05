import torch
from tqdm import tqdm
import os
from config import Config
from loss_function import bce_loss
from pathlib import Path
import numpy as np
import wandb

def train(net, training_loader, validation_loader, optimizer, epochs, start_epoch,  validate_every, output_dir):
    device = Config.get('device')
    best_accuracy = 0
    batch_accuracy = []
    batch_accuracy_row = []
    batch_accuracy_col = []

    artifact = wandb.Artifact(
    name='table-segmentation-static-images-11-7', 
    type='model'
    )   

    for epoch in range(start_epoch, epochs + 1):
        net.train()
        epoch_loss = 0
        correct_count = 0
        count = 0
        for i, data in tqdm(enumerate(training_loader)):
            img = data[0].to(device)
            row_labels = data[1].to(device)
            col_labels = data[2].to(device)

            labels = (row_labels, col_labels)

            #-------*******fix this
            try:
                pred_labels = net(img)
            except:
                print('Error****')
                continue
            r, c = pred_labels
            loss = bce_loss(pred_labels, labels)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #times = 1
            #correct_count += (torch.sum((pred_labels[0] > 0.5).type(torch.IntTensor) == labels[0][0].repeat(times,1).type(torch.IntTensor)).item() + torch.sum((pred_labels[1] > 0.5).type(torch.IntTensor) == labels[1][0].repeat(times,1).type(torch.IntTensor)).item())

            #count += labels[0].view(-1).size()[0] * times + labels[1].view(-1).size()[0] * times
            torch.cuda.empty_cache()
            pred_row_labels, pred_col_labels = pred_labels
            c5 = pred_col_labels[2]
            r5 = pred_row_labels[2]
            
            r5 = r5[-1] > 0.5
            c5 = c5[-1] > 0.5

            #-------------------------col---------------------------------------------
            predicted_col_labels = c5.type(torch.IntTensor).cpu() #.detach().numpy()
            target_col_labels = col_labels.cpu() #.numpy()

            confusion_vector_col = predicted_col_labels / target_col_labels

            true_positives_col = torch.sum(confusion_vector_col == 1).item()
            false_positives_col = torch.sum(confusion_vector_col == float('inf')).item()
            true_negatives_col = torch.sum(torch.isnan(confusion_vector_col)).item()
            false_negatives_col = torch.sum(confusion_vector_col == 0).item()
            accuracy_col = (true_positives_col + true_negatives_col) / (true_positives_col + false_positives_col + true_negatives_col + false_negatives_col)
            batch_accuracy_col.append(accuracy_col)

            #----------------------------row------------------------------------------
            predicted_row_labels = r5.type(torch.IntTensor).cpu() #.detach().numpy()
            target_row_labels = row_labels.cpu() #.numpy()

            confusion_vector_row = predicted_row_labels / target_row_labels


            true_positives_row = torch.sum(confusion_vector_row == 1).item()
            false_positives_row = torch.sum(confusion_vector_row == float('inf')).item()
            true_negatives_row = torch.sum(torch.isnan(confusion_vector_row)).item()
            false_negatives_row = torch.sum(confusion_vector_row == 0).item()

            accuracy_row = (true_positives_row + true_negatives_row) / (true_positives_row + false_positives_row + true_negatives_row + false_negatives_row)
            batch_accuracy_row.append(accuracy_row)

            #-----------------------------table---------------------------------------
            true_positives = true_positives_row + true_positives_col
            false_positives = false_positives_row + false_positives_col
            true_negatives = true_negatives_row + true_negatives_col
            false_negatives = false_negatives_row + false_negatives_col

            accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
            batch_accuracy.append(accuracy)



        accuracy = np.mean(batch_accuracy)
        print('Epoch {0} finished ! Loss: {1}'.format(epoch, epoch_loss / (i + 1)))
        print('Accuracy:  ', np.mean(batch_accuracy))
        print('Row Accuracy:  ', np.mean(batch_accuracy_row))
        print('Column Accuracy:  ', np.mean(batch_accuracy_col))

        train_metrics = {'train/acc' : np.mean(batch_accuracy), 'train/row_acc' : np.mean(batch_accuracy_row), 'train/col_acc' : np.mean(batch_accuracy_col),
                         'train/loss' : epoch_loss / (i+1)}
        wandb.log(train_metrics)

        if epoch % validate_every == 0:
            validation_loss, validation_accuracy = validation(net, validation_loader)
            #if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
        #saving every epoch checkpoint
        models_dir = Path(output_dir / 'checkpoints')
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        checkpoint_file = 'deep_table_splitting--epoch--' + str(epoch) + '.pth.tar'
        checkpoint_path = Path(models_dir / checkpoint_file)
        torch.save({'epoch': epoch,
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }, str(checkpoint_path))
        artifact.add_file(local_path=str(checkpoint_path))


def validation(net, validation_loader):
    device = Config.get('device')
    net.eval()
    epoch_loss = 0
    correct_count = 0
    count = 0
    times = 1

    batch_accuracy = []
    batch_precision = []
    batch_recall = []
    batch_f1 = []

    batch_accuracy_row = []
    batch_precision_row = []
    batch_recall_row = []
    batch_f1_row = []

    batch_accuracy_col = []
    batch_precision_col = []
    batch_recall_col = []
    batch_f1_col = []

    for i, data in tqdm(enumerate(validation_loader)):
        img = data[0].to(device)
        row_labels = data[1].to(device)
        col_labels = data[2].to(device)

        labels = (row_labels, col_labels)

        #-------*******fix this
        try:
            pred_labels = net(img)
        except:
            print('Validation Error***')
            continue
        loss = bce_loss(pred_labels, labels)

        epoch_loss += loss.item()

        #correct_count += (torch.sum((pred_labels[2] > 0.5).type(torch.IntTensor) == labels[1][0].repeat(times,1).type(torch.IntTensor)).item())

        #count += labels[1].view(-1).size()[0] * times
        torch.cuda.empty_cache()
        
        pred_row_labels, pred_col_labels = pred_labels
        #evaluation metrics
        c5 = pred_col_labels[2]
        r5 = pred_row_labels[2]
        
        r5 = r5[-1] > 0.5
        c5 = c5[-1] > 0.5

        #---------------------------------col-------------------------------------
        predicted_col_labels = c5.type(torch.IntTensor).cpu() #.detach().numpy()
        target_col_labels = col_labels.cpu() #.numpy()

        confusion_vector_col = predicted_col_labels / target_col_labels


        true_positives_col = torch.sum(confusion_vector_col == 1).item()
        false_positives_col = torch.sum(confusion_vector_col == float('inf')).item()
        true_negatives_col = torch.sum(torch.isnan(confusion_vector_col)).item()
        false_negatives_col = torch.sum(confusion_vector_col == 0).item()

        accuracy_col = (true_positives_col + true_negatives_col) / (true_positives_col + false_positives_col + true_negatives_col + false_negatives_col)
        try:
            precision_col = true_positives_col / (true_positives_col + false_positives_col)
        except ZeroDivisionError:
            precision_col = 0
        
        try:
            recall_col = true_positives_col / (true_positives_col + false_negatives_col)
        except ZeroDivisionError:
            recall = 0

        try:
            f1_col = 2 * (precision_col * recall_col) / (precision_col + recall_col)
        except ZeroDivisionError:
            f1_col = 0
        
        batch_accuracy_col.append(accuracy_col)
        batch_precision_col.append(precision_col)
        batch_recall_col.append(recall_col)
        batch_f1_col.append(f1_col)

        #------------------------------------row----------------------------------

        predicted_row_labels = r5.type(torch.IntTensor).cpu() #.detach().numpy()
        target_row_labels = row_labels.cpu() #.numpy()

        confusion_vector_row = predicted_row_labels / target_row_labels


        true_positives_row = torch.sum(confusion_vector_row == 1).item()
        false_positives_row = torch.sum(confusion_vector_row == float('inf')).item()
        true_negatives_row = torch.sum(torch.isnan(confusion_vector_row)).item()
        false_negatives_row = torch.sum(confusion_vector_row == 0).item()

        accuracy_row = (true_positives_row + true_negatives_row) / (true_positives_row + false_positives_row + true_negatives_row + false_negatives_row)
        try:
            precision_row = true_positives_row / (true_positives_row + false_positives_row)
        except ZeroDivisionError:
            precision_row = 0
        
        try:
            recall_row = true_positives_row / (true_positives_row + false_negatives_row)
        except ZeroDivisionError:
            recall_row = 0

        try:
            f1_row = 2 * (precision_row * recall_row) / (precision_row + recall_row)
        except ZeroDivisionError:
            f1_row = 0
        
        batch_accuracy_row.append(accuracy_row)
        batch_precision_row.append(precision_row)
        batch_recall_row.append(recall_row)
        batch_f1_row.append(f1_row)

        #---------------------------table--------------------------------------
        true_positives = true_positives_row + true_positives_col
        false_positives = false_positives_row + false_positives_col
        true_negatives = true_negatives_row + true_negatives_col
        false_negatives = false_negatives_row + false_negatives_col

        accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        try:
            precision = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            precision = 0
        
        try:
            recall = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        
        batch_accuracy.append(accuracy)
        batch_precision.append(precision)
        batch_recall.append(recall)
        batch_f1.append(f1)


    total_loss = epoch_loss / (i + 1)

    print('Validation finished ! Loss: {0}'.format(epoch_loss / (i + 1)))
    print('Accuracy:  ', np.mean(batch_accuracy))
    print('Precision: ', np.mean(batch_precision))
    print('Recall:    ', np.mean(batch_recall))
    print('F1:        ', np.mean(batch_f1))

    print('Row Accuracy:  ', np.mean(batch_accuracy_row))
    print('Row Precision: ', np.mean(batch_precision_row))
    print('Row Recall:    ', np.mean(batch_recall_row))
    print('Row F1:        ', np.mean(batch_f1_row))

    print('Column Accuracy:  ', np.mean(batch_accuracy_col))
    print('Column Precision: ', np.mean(batch_precision_col))
    print('Column Recall:    ', np.mean(batch_recall_col))
    print('Column F1:        ', np.mean(batch_f1_col))
    accuracy = np.mean(batch_accuracy)

    log_metrics = {'val/acc' : np.mean(batch_accuracy), 'val/row_acc' : np.mean(np.mean(batch_accuracy_row)), 'val/col_acc' : np.mean(batch_accuracy_col),
                   'val/precision' : np.mean(batch_precision), 'val/row_precision' : np.mean(batch_precision_row), 'val/col_precision' : np.mean(batch_precision_col),
                   'val/recall' : np.mean(batch_recall), 'val/row_recall' : np.mean(batch_recall_row), 'val/col_recall' : np.mean(batch_recall_col),
                   'val/f1' : np.mean(batch_f1), 'val/row_f1' : np.mean(batch_f1_row), 'val/col_f1' : np.mean(batch_f1_col),
                   'val/loss' : epoch_loss / (i+1)}
    wandb.log(log_metrics)

    return total_loss, accuracy

def testing(net, testing_loader):
    device = Config.get('device')
    net.eval()
    epoch_loss = 0
    correct_count = 0
    count = 0
    times = 1

    batch_accuracy = []
    batch_precision = []
    batch_recall = []
    batch_f1 = []

    batch_accuracy_row = []
    batch_precision_row = []
    batch_recall_row = []
    batch_f1_row = []

    batch_accuracy_col = []
    batch_precision_col = []
    batch_recall_col = []
    batch_f1_col = []

    for i, data in tqdm(enumerate(testing_loader)):
        img = data[0].to(device)
        row_labels = data[1].to(device)
        col_labels = data[2].to(device)

        labels = (row_labels, col_labels)

        pred_labels = net(img)
        loss = bce_loss(pred_labels, labels)

        epoch_loss += loss.item()

        #correct_count += (torch.sum((pred_labels[2] > 0.5).type(torch.IntTensor) == labels[1][0].repeat(times,1).type(torch.IntTensor)).item())

        #count += labels[1].view(-1).size()[0] * times
        torch.cuda.empty_cache()
        
        pred_row_labels, pred_col_labels = pred_labels
        #evaluation metrics
        c5 = pred_col_labels[2]
        r5 = pred_row_labels[2]
        
        r5 = r5[-1] > 0.5
        c5 = c5[-1] > 0.5

        #---------------------------------col-------------------------------------
        predicted_col_labels = c5.type(torch.IntTensor).cpu() #.detach().numpy()
        target_col_labels = col_labels.cpu() #.numpy()

        confusion_vector_col = predicted_col_labels / target_col_labels


        true_positives_col = torch.sum(confusion_vector_col == 1).item()
        false_positives_col = torch.sum(confusion_vector_col == float('inf')).item()
        true_negatives_col = torch.sum(torch.isnan(confusion_vector_col)).item()
        false_negatives_col = torch.sum(confusion_vector_col == 0).item()

        accuracy_col = (true_positives_col + true_negatives_col) / (true_positives_col + false_positives_col + true_negatives_col + false_negatives_col)
        try:
            precision_col = true_positives_col / (true_positives_col + false_positives_col)
        except ZeroDivisionError:
            precision_col = 0
        
        try:
            recall_col = true_positives_col / (true_positives_col + false_negatives_col)
        except ZeroDivisionError:
            recall = 0

        try:
            f1_col = 2 * (precision_col * recall_col) / (precision_col + recall_col)
        except ZeroDivisionError:
            f1_col = 0
        
        batch_accuracy_col.append(accuracy_col)
        batch_precision_col.append(precision_col)
        batch_recall_col.append(recall_col)
        batch_f1_col.append(f1_col)

        #------------------------------------row----------------------------------

        predicted_row_labels = r5.type(torch.IntTensor).cpu() #.detach().numpy()
        target_row_labels = row_labels.cpu() #.numpy()

        confusion_vector_row = predicted_row_labels / target_row_labels


        true_positives_row = torch.sum(confusion_vector_row == 1).item()
        false_positives_row = torch.sum(confusion_vector_row == float('inf')).item()
        true_negatives_row = torch.sum(torch.isnan(confusion_vector_row)).item()
        false_negatives_row = torch.sum(confusion_vector_row == 0).item()

        accuracy_row = (true_positives_row + true_negatives_row) / (true_positives_row + false_positives_row + true_negatives_row + false_negatives_row)
        try:
            precision_row = true_positives_row / (true_positives_row + false_positives_row)
        except ZeroDivisionError:
            precision_row = 0
        
        try:
            recall_row = true_positives_row / (true_positives_row + false_negatives_row)
        except ZeroDivisionError:
            recall_row = 0

        try:
            f1_row = 2 * (precision_row * recall_row) / (precision_row + recall_row)
        except ZeroDivisionError:
            f1_row = 0
        
        batch_accuracy_row.append(accuracy_row)
        batch_precision_row.append(precision_row)
        batch_recall_row.append(recall_row)
        batch_f1_row.append(f1_row)

        #---------------------------table--------------------------------------
        true_positives = true_positives_row + true_positives_col
        false_positives = false_positives_row + false_positives_col
        true_negatives = true_negatives_row + true_negatives_col
        false_negatives = false_negatives_row + false_negatives_col

        accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
        try:
            precision = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            precision = 0
        
        try:
            recall = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            recall = 0

        try:
            f1 = 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        
        batch_accuracy.append(accuracy)
        batch_precision.append(precision)
        batch_recall.append(recall)
        batch_f1.append(f1)


    total_loss = epoch_loss / (i + 1)
    print('Testing finished ! Loss: {0}'.format(epoch_loss / (i + 1)))
    print('Accuracy:  ', np.mean(batch_accuracy))
    print('Precision: ', np.mean(batch_precision))
    print('Recall:    ', np.mean(batch_recall))
    print('F1:        ', np.mean(batch_f1))

    print('Row Accuracy:  ', np.mean(batch_accuracy_row))
    print('Row Precision: ', np.mean(batch_precision_row))
    print('Row Recall:    ', np.mean(batch_recall_row))
    print('Row F1:        ', np.mean(batch_f1_row))

    print('Column Accuracy:  ', np.mean(batch_accuracy_col))
    print('Column Precision: ', np.mean(batch_precision_col))
    print('Column Recall:    ', np.mean(batch_recall_col))
    print('Column F1:        ', np.mean(batch_f1_col))
    accuracy = np.mean(batch_accuracy)
    return total_loss, accuracy
