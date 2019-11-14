class Trainer:
    def __init__(self, args):
        self.args = args
        self.train_loader, self.val_loader, self.test_loader = split_data_EAN(cls_file_name, KG_file_name, context_file_name, labels_file_name, val_ratio)

    def train(self, num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir,
              val_every, device, epochs=0, avg_val_loss_list=[], avg_train_loss_list=[]):
        best_loss = 9999
        avg_val_loss_list = []
        avg_train_loss_list = []

        for epoch in range(epochs, num_epochs):
            temp_epoch_loss = []
            for step, (cls, kg_context, y) in enumerate(train_loader):
                cls, kg_context, y = cls.to(device), kg_context.to(device), y.to(device)

                outputs = model(cls, kg_context)
                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (step + 1) % 25 == 0:
                    print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, step + 1, len(train_loader), loss.item()))

                temp_epoch_loss.append(loss.item())

            avg_train_loss = sum(temp_epoch_loss) / len(temp_epoch_loss)

            if (epoch + 1) % val_every == 0:  # Compare and save the best model
                avg_loss = self.validation(epoch + 1, model, val_loader, criterion, device)

                avg_train_loss_list.append(avg_train_loss)
                avg_val_loss_list.append(avg_loss)

                if avg_loss < best_loss:
                    print("Best performance at epoch: {}".format(epoch + 1))
                    print("Save model in", saved_dir)
                    best_loss = avg_loss
                    save_model(model, optimizer, epoch, avg_train_loss_list, avg_val_loss_list, saved_dir,
                               file_name='with_entity_2_model_bs64_lr1e-5_nl9_h12.tar')

        plot_results(np.arange(0, num_epochs), avg_train_loss_list, avg_val_loss_list)

    def validation(self, epoch, model, data_loader, criterion, device):
        print('Start validation #{}'.format(epoch))
        model.eval()
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            for step, (cls, kg_context, y) in enumerate(data_loader):
                cls, kg_context, y = cls.to(device), kg_context.to(device), y.to(device)

                outputs = model(cls, kg_context)
                loss = criterion(outputs, y)
                total_loss += loss
                cnt += 1
            avrg_loss = total_loss / cnt
            print('Validation #{}  Average Loss: {:.4f}'.format(epoch, avrg_loss))
        model.train()
        return avrg_loss

    def test(self, model, data_loader, criterion, device):
        print('Start test..')
        model.eval()
        with torch.no_grad():
            total_loss = 0
            cnt = 0
            total_acc = []
            for step, (cls, kg_context, y) in enumerate(data_loader):
                cls, kg_context, y = cls.to(device), kg_context.to(device), y.to(device)

                outputs = model(cls, kg_context)
                loss = criterion(outputs, y)

                total_loss += loss
                cnt += 1

                if (step + 1) % 50 == 0:
                    print('Step: ', step, 'Loss: ', loss.item(), 'Accuracy: ', accuracy(outputs, y))
                total_acc.append(accuracy(outputs, y))
            avrg_loss = total_loss / cnt
            print('Test Average Loss: {:.4f}'.format(avrg_loss))
            print('Test Accuracy: {}'.format(sum(total_acc, 0.0) / len(total_acc)))

    def save_model(model, optimizer, epoch, train_loss, validation_loss, saved_dir, file_name='best_model_bs64.pt'):
        os.makedirs(saved_dir, exist_ok=True)
        check_point = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'total_train_loss': train_loss,
            'total_validation_loss': validation_loss
        }
        output_path = os.path.join(saved_dir, file_name)
        torch.save(check_point, output_path)

    def accuracy(pred, target):
        pred_y = pred >= 0.5
        #     print(pred_y.dtype)
        num_correct = target.eq(pred_y.float()).sum()
        #     print(num_correct)
        accuracy = (num_correct.item() * 100.0 / len(target))
        return accuracy

    def plot_results(x, y, z):
        """
        plot the results
        """

        fig = plt.figure(figsize=(9, 6))
        plt.plot(x, y, z)
        plt.xlabel('Number of steps')
        plt.ylabel('loss')
        plt.legend(('train_loss', 'validation_loss'))
        plt.title("Loss over epoch")
        plt.show()