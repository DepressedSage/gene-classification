import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from optimizer import ScheduledOptim
import tqdm


# We will define the BERTTrainer class which will be used to train the BERT model.
class BERTTrainer:
    def __init__(
            self,
            model,
            train_dataloader,
            val_dataloader=None,
            test_dataloader=None,
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            warmup_steps=10000,
            patience=3,
            log_freq=10,
            checkpoint_dir='../checkpoints',
            load_checkpoint='None',
            load_pretrained=False,
            device='cuda'
            ):

        self.device = device
        self.model = model
        # self.classiffier = classifier
        self.train_data = train_dataloader
        self.val_data = val_dataloader
        self.best_val_loss = float("inf")
        self.test_data = test_dataloader
        self.patience = patience

        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

        self.counter = 0
        # if load_pretrained:
        #    self.model = self.load_checkpoint(load_checkpoint)
        self.model.to(self.device)

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(
                self.optim, self.model.embed_size, n_warmup_steps=warmup_steps
                )

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = torch.nn.NLLLoss(ignore_index=0)
        self.log_freq = log_freq
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epochs, dry_run=False):
        os.makedirs(f"{self.checkpoint_dir}/epochs", exist_ok=True)
        if self.val_data is not None:
            for epoch in range(1, epochs + 1):
                self.iteration(epoch, self.train_data, mode="train")
                val_loss = self.iteration(epoch, self.val_data, mode="validate")
                if val_loss < self.best_val_loss:
                    print(f'Validation loss decreased from {self.best_val_loss} to {val_loss}.')
                    self.best_val_loss = val_loss
                    self.counter = 0
                else:
                    self.counter += 1
                    print(f'Early stopping counter: {self.counter} out of {self.patience}.')
                    print(f'Validation loss did not decrease from {self.best_val_loss}.')
                    if self.counter > self.patience:
                        print(f'Early stopping after {epoch} epochs.')
                        break
                if not dry_run and epoch % 5 == 0:
                    self.save_checkpoint(f"epochs/epoch_{epoch}")
        else:
            for epoch in range(1, epochs + 1):
                self.iteration(epoch, self.train_data, mode="train")
                if not dry_run and epoch % 5 == 0:
                    self.save_checkpoint(f"epochs/epoch_{epoch}")

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, mode="train"):

        avg_loss = 0.0
        avg_classifier_loss = 0
        avg_mask_loss = 0
        total_correct = 0
        total_element = 0

        torch.autograd.set_detect_anomaly(True)
        # progress bar
        data_iter = tqdm.tqdm(
                enumerate(data_loader),
                desc="EP_%s:%d" % (mode, epoch),
                total=len(data_loader),
                bar_format="{l_bar}{r_bar}"
                )

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the masked_lm model
            classifier_output, mask_lm_output = self.model.forward(data["bert_input"], data["segment_label"])

            # 2. NLLLoss of predicting masked token word
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["bert_label"])

            classifier_loss = F.binary_cross_entropy_with_logits(
                classifier_output, data["sequence_label"].float())
            # classifier_loss = self.criterion(classifier_output, data["sequence_label"])

            if torch.isnan(mask_loss):
                mask_loss = avg_loss/(i+1)
                # Additional debugging or error handling can be added here

            # Check if classifier_loss is NaN
            if torch.isnan(classifier_loss):
                classifier_loss = avg_loss/(i+1)

            if isinstance(mask_loss, torch.Tensor):
                avg_mask_loss += mask_loss.item()
            else:
                # If loss is already a float, just accumulate it directly
                avg_mask_loss += avg_mask_loss

            if isinstance(classifier_loss, torch.Tensor):
                avg_classifier_loss += classifier_loss.item()
            else:
                # If loss is already a float, just accumulate it directly
                avg_classifier_loss += avg_classifier_loss

            loss = mask_loss + classifier_loss

            # 3. backward and optimization only in train
            if mode == "train":
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            binary_predictions = (classifier_output>= 0.5).int()

            # Count correct predictions
            correct_classified = (binary_predictions == data["sequence_label"]).sum().item()
            # correct_classified = (classifier_output.argmax(1) == data["sequence_label"]).sum().item()
            if isinstance(loss, torch.Tensor):
                avg_loss += loss.item()
            else:
                # If loss is already a float, just accumulate it directly
                avg_loss += avg_loss
            total_correct += correct_classified
            total_element += data["sequence_label"].nelement()

            post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss/(i + 1),
                    "avg_mask_loss": avg_mask_loss/(i + 1),
                    "avg_classifier_loss": avg_classifier_loss/(i + 1),
                    "avg_acc": total_correct/total_element,
                    "loss": loss.item()
                    }

            if i % self.log_freq == 0:
                data_iter.set_postfix(post_fix)  # Update progress bar with current metrics
        # print(f"EP{epoch}, {mode}: avg_loss={avg_loss / len(data_iter)}")
        if mode == "validate":
            return avg_loss / len(data_iter)

    def save_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name + ".pth")
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            # Add any other information you want to save in the checkpoint
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_name):
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name + ".pth")
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return self.model
