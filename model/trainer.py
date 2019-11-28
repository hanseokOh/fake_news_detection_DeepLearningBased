import numpy as np
from dataLoader import split_data
from utils import save_model, plot_results, accuracy, f1_measure

import torch

class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.val_loader, self.test_loader = split_data(args.sent_pad_path, args.label_path)

    def train(self, num_epochs, model, saved_dir, device,criterion, optimizer, val_every):
        if criterion is None:
            criterion = torch.nn.BCELoss()
        if optimizer is None:
            optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-5)
            # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

        best_loss = 9999
        avg_val_loss_list = []
        avg_train_loss_list = []

        for epoch in range(num_epochs):
            temp_epoch_loss = []
            for step, (sequence, target) in enumerate(self.train_loader):
                sequence = sequence
                target = target
                sequence, target = sequence.to(device), target.to(device)

                outputs, _ = model(sequence)
                loss = criterion(outputs, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % 25 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, step + 1, len(self.train_loader), loss.item()))

                temp_epoch_loss.append(loss.item())

            avg_train_loss = sum(temp_epoch_loss) / len(temp_epoch_loss)

            if (epoch + 1) % val_every == 0:  # Compare and save the best model
                avg_loss = self.validation(epoch + 1, model, self.val_loader, criterion, device)

                if avg_loss < best_loss:
                    print("Best performance at epoch: {}".format(epoch + 1))
                    print("Save model in", saved_dir)
                    best_loss = avg_loss
                    #                 save_model(model, optimizer, epoch, best_loss, saved_dir)

                    if len(avg_val_loss_list) == 0:
                        save_model(model, optimizer, epoch, avg_train_loss, best_loss, saved_dir)

                    else:
                        save_model(model, optimizer, epoch, avg_train_loss_list, avg_val_loss_list, saved_dir)

                avg_train_loss_list.append(avg_train_loss)
                avg_val_loss_list.append(avg_loss)

        plot_results(np.arange(0, num_epochs), avg_train_loss_list, avg_val_loss_list)


    def validation(self, epoch, model, device, criterion=None):
        if criterion is None:
            criterion = torch.nn.BCELoss()

        print('Start validation #{}'.format(epoch))
        model.eval()
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            for step, (sequence, target) in enumerate(self.val_loader):
                sequence = sequence #.type(torch.float32)
                target = target #.type(torch.float32)
                sequence, target = sequence.to(device), target.to(device)
                outputs, _ = model(sequence)
                loss = criterion(outputs, target)
                total_loss += loss
                cnt += 1
            avrg_loss = total_loss / cnt
            print('Validation #{}  Average Loss: {:.4f}'.format(epoch, avrg_loss))
        model.train()
        return avrg_loss


    def test(self, model, device, criterion=None):
        if criterion is None:
            criterion = torch.nn.BCELoss()
        print('Start test..')
        model.eval()
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            total_f1 = []
            total_acc = []
            for step, (sequence, target) in enumerate(self.test_loader):
                sequence = sequence  # .type(torch.float32)
                target = target  # .type(torch.float32)
                sequence, target = sequence.to(device), target.to(device)

                outputs, _ = model(sequence)
                loss = criterion(outputs, target)

                total_loss += loss
                cnt += 1

                if (step + 1) % 50 == 0:
                    print('Step: ', step, 'Loss: ', loss.item(), 'Accuracy: ', accuracy(outputs, target))
                total_f1.append(f1_measure(outputs, target))
                total_acc.append(accuracy(outputs, target))

            avrg_loss = total_loss / cnt
            print('Test  Average Loss: {:.4f}'.format(avrg_loss))
            print("Average F1: {}".format(sum(total_f1)/len(total_f1)))
            print("Average Acc: {}".format(sum(total_acc)/len(total_acc)))
