import torch
from tqdm import tqdm

from config import Config
from loss_function import bce_loss

def train(net, training_loader, validation_loader, optimizer, epochs, start_epoch,  validate_every):
    device = Config.get('device')
    best_accuracy = 0
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

            pred_labels = net(img)
            #r, c = pred_labels
            loss = bce_loss(pred_labels, labels)

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            times = 1
            correct_count += (torch.sum((pred_labels[2] > 0.5).type(torch.IntTensor) == labels[1][0].repeat(times,1).type(torch.IntTensor)).item())

            count += labels[1].view(-1).size()[0] * times
            torch.cuda.empty_cache()

        accuracy = correct_count / (count)
        print('Epoch {0} finished ! Loss: {1} , Accuracy: {2}'.format(epoch, epoch_loss / (i + 1),accuracy))
        validation_loss, validation_accuracy = validation(net, validation_loader)
        if validation_accuracy > best_accuracy:
            checkpoint_file = '/gdrive/My Drive/deep-splitting-merging/model/checkpoint.deep_column_splitting.pth.tar'
            torch.save({'epoch': epoch,
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        }, str(checkpoint_file))


def validation(net, validation_loader):
    net.eval()
    device = Config.get('device')
    epoch_loss = 0
    correct_count = 0
    count = 0
    times = 1
    for i, data in tqdm(enumerate(validation_loader)):
        img = data[0].to(device)
        row_labels = data[1].to(device)
        col_labels = data[2].to(device)

        labels = (row_labels, col_labels)

        pred_labels = net(img)
        loss = bce_loss(pred_labels, labels)

        epoch_loss += loss.item()

        correct_count += (torch.sum((pred_labels[2] > 0.5).type(torch.IntTensor) == labels[1][0].repeat(times,1).type(torch.IntTensor)).item())

        count += labels[1].view(-1).size()[0] * times
        torch.cuda.empty_cache()

    accuracy = correct_count / (count)
    total_loss = epoch_loss / (i + 1)
    print('Validation finished ! Loss: {0} , Accuracy: {1}'.format(epoch_loss / (i + 1), accuracy))
    
    return total_loss, accuracy


def testing(net, testing_loader):
    net.eval()
    device = Config.get('device')
    epoch_loss = 0
    correct_count = 0
    count = 0
    times = 1
    for i, data in tqdm(enumerate(testing_loader)):
        img = data[0].to(device)
        row_labels = data[1].to(device)
        col_labels = data[2].to(device)

        labels = (row_labels, col_labels)

        pred_labels = net(img)
        loss = bce_loss(pred_labels, labels)

        epoch_loss += loss.item()

        correct_count += (torch.sum((pred_labels[2] > 0.5).type(torch.IntTensor) == labels[1][0].repeat(times,1).type(torch.IntTensor)).item())

        count += labels[1].view(-1).size()[0] * times
        torch.cuda.empty_cache()

    accuracy = correct_count / (count)
    total_loss = epoch_loss / (i + 1)
    print('Validation finished ! Loss: {0} , Accuracy: {1}'.format(epoch_loss / (i + 1), accuracy))
    
    return total_loss, accuracy