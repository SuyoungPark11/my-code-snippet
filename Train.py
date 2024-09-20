import os
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from tqdm.auto import tqdm
import matplotlib.pyplot as plt # type: ignore

class Loss(nn.Module):
    def __init__(
        self,
        loss_fn: str
    ):
        super(Loss, self).__init__()
        loss_dict = {
            "L1" : nn.L1Loss,
            "MSE" : nn.MSELoss,
            "CE" : nn.CrossEntropyLoss,
            "BCE" : nn.BCELoss
        }
        self.loss_fn = loss_dict[loss_fn]

    def forward(
        self, 
        outputs: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:    
        return self.loss_fn(outputs, targets)
    
class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        device: torch.device, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        loss_fn: torch.nn.modules.loss._Loss, 
        epochs: int,
        result_path: str,
        patience: int = 5
    ):
        self.model = model 
        self.device = device  
        self.train_loader = train_loader  
        self.val_loader = val_loader  
        self.optimizer = optimizer  
        self.scheduler = scheduler 
        self.loss_fn = loss_fn  
        self.epochs = epochs  
        self.result_path = result_path        # model save path
        self.best_models = []                 # top 3 model information 
        self.lowest_train_loss = float('inf') 
        self.lowest_val_loss = float('inf')
        self.patience = patience              # Early Stopping patience
        self.early_stop_counter = 0

        # To save loss, accuracy
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def accuracy_fn(self, outputs, targets):
        """
        Calculate accuracy
        """
        _, preds = torch.max(outputs, 1)
        corrects = torch.sum(preds == targets)
        accuracy = corrects.double() / len(targets)
        return accuracy.item()
    
    def save_model(self, epoch, loss):
        """
        Save model  
        - process : make directory -> save model at current epoch -> manage top-3 model -> save best model 
        """
        os.makedirs(self.result_path, exist_ok=True)

        current_model_path = os.path.join(self.result_path, f'model_epoch_{epoch}_loss_{loss:.4f}.pt')
        torch.save(self.model.state_dict(), current_model_path)

        self.best_models.append((loss, epoch, current_model_path))
        self.best_models.sort()
        if len(self.best_models) > 3:
            _, _, path_to_remove = self.best_models.pop(-1)  # remove the highest loss model 
            if os.path.exists(path_to_remove):
                os.remove(path_to_remove)

        if loss < self.lowest_train_loss:
            self.lowest_train_loss = loss
            best_model_path = os.path.join(self.result_path, 'best_model.pt')
            torch.save(self.model.state_dict(), best_model_path)
            print(f"Save {epoch}epoch result. Loss = {loss:.4f}")

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        print(f"Model loaded from {model_path}")

    def train_epoch(self) -> (float, float): # type: ignore
        self.model.train()
        total_loss = 0.0
        corrects = 0
        total = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, targets in progress_bar:
            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            corrects += torch.sum(torch.max(outputs, 1)[1] == targets).item()
            total += targets.size(0)
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = corrects / total
        return avg_loss, accuracy

    def validate(self) -> (float, float): # type: ignore
        self.model.eval()
        total_loss = 0.0
        corrects = 0
        total = 0

        progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, targets in progress_bar:
                images, targets = images.to(self.device), targets.to(self.device)
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                corrects += torch.sum(torch.max(outputs, 1)[1] == targets).item()
                total += targets.size(0)
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)
        accuracy = corrects / total
        return avg_loss, accuracy
    
    def train(self, load_model: bool = False, model_path: str = None) -> None:
        """
        Mange all training process 
        """
        if load_model and model_path:
            self.load_model(model_path)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.save_model(epoch, val_loss)

            if val_loss < self.lowest_val_loss:
                self.lowest_val_loss = val_loss
                self.early_stop_counter = 0  
                print(f"Lowest loss updated to {self.lowest_val_loss:.4f}. Early stop counter reset to 0.")
            else:
                self.early_stop_counter += 1  
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}.")
                    break

            self.scheduler.step()

            # Plot a result at every epoch 
            self.plot_results()

        # Plot the final result
        self.plot_results()

    def plot_results(self):
        """학습 결과 시각화 (Loss & Accuracy)"""
        epochs_range = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(14, 10))

        # Loss 시각화
        plt.subplot(2, 1, 1)
        plt.plot(epochs_range, self.train_losses, label="Train Loss", color='blue')
        plt.plot(epochs_range, self.val_losses, label="Validation Loss", color='red')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.title("Train & Validation Loss")

        # Accuracy 시각화
        plt.subplot(2, 1, 2)
        plt.plot(epochs_range, self.train_accuracies, label="Train Accuracy", color='green')
        plt.plot(epochs_range, self.val_accuracies, label="Validation Accuracy", color='orange')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.title("Train & Validation Accuracy")

        plt.tight_layout()
        plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = Loss(loss_fn="MSE")
optimizer = optim.Adam(
    model.parameters(), 
    lr=0.001
)
# Scheduling 
scheduler_step_size = 30  
scheduler_gamma = 0.1  
steps_per_epoch = len(train_loader)
epochs_per_lr_decay = 2
scheduler_step_size = steps_per_epoch * epochs_per_lr_decay

scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=scheduler_step_size, 
    gamma=scheduler_gamma
)

trainer = Trainer(
    model=model, 
    device=device, 
    train_loader=train_loader,
    val_loader=val_loader, 
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn, 
    epochs=5,
    result_path="save_result_path"
)
trainer.train()