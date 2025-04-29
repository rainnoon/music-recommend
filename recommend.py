

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import os
from itertools import product
from collections import namedtuple, OrderedDict
from IPython.display import display, clear_output
import time
import json
from torchsummary import summary
import matplotlib.pyplot as plt
import random

# Print settings
torch.set_printoptions(linewidth=120)


ANNOTATIONS_FILE = "./GTZAN_TEST/features_30_sec_test.csv"
dataframe = pd.read_csv(ANNOTATIONS_FILE)
labels = set()
for row in range(len(dataframe)):
    labels.add(dataframe.iloc[row, -1])
labels_list = []
for label in labels:
    labels_list.append(label)
sorted_labels = sorted(labels_list)
mapping = {}
for index, label in enumerate(sorted_labels):
    mapping[label] = index
dataframe["num_label"] = dataframe["label"]
new_dataframe = dataframe.replace({"num_label": mapping})
new_dataframe.to_csv("features_30_sec_test_final.csv")

# Training set label processing
ANNOTATIONS_FILE = "./GTZAN/features_30_sec.csv"
dataframe = pd.read_csv(ANNOTATIONS_FILE)
labels = set()
for row in range(len(dataframe)):
    labels.add(dataframe.iloc[row, -1])
labels_list = []
for label in labels:
    labels_list.append(label)
sorted_labels = sorted(labels_list)
mapping = {}
for index, label in enumerate(sorted_labels):
    mapping[label] = index
dataframe["num_label"] = dataframe["label"]
new_dataframe = dataframe.replace({"num_label": mapping})
new_dataframe.to_csv("features_30_sec_final.csv")

# =====================
# Hyperparameter management class
# =====================
class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for element in product(*params.values()):
            runs.append(Run(*element))
        return runs

# =====================
# Training process management class
# =====================
class RunManager():
    def __init__(self):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_correct_num = 0
        self.epoch_start_time = None
        self.test_epoch_count = 0
        self.test_epoch_loss = 0
        self.test_epoch_correct_num = 0
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        self.network = None
        self.loader = None
        self.tb = None
    def begin_run(self, run, network, loader, test_loader):
        self.run_start_time = time.time()
        self.run_params = run
        self.run_count += 1
        self.network = network
        self.loader = loader
        self.test_loader = test_loader
        self.tb = SummaryWriter(comment=f'-{run}')
        signal, sr, address = next(iter(self.loader))
        self.tb.add_graph(
            self.network,
            signal.to(run.device)
        )
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0
        self.test_epoch_count = 0
    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count += 1
        self.epoch_loss = 0
        self.epoch_correct_num = 0
        self.test_epoch_count += 1
        self.test_epoch_loss = 0
        self.test_epoch_correct_num = 0
    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        loss = self.epoch_loss / len(self.loader.dataset)
        accuracy = self.epoch_correct_num / len(self.loader.dataset)
        print(f'Accuracy: {self.epoch_correct_num} / {len(self.loader.dataset)}')
        test_loss = self.test_epoch_loss / len(self.test_loader.dataset)
        test_accuracy = self.test_epoch_correct_num / len(self.test_loader.dataset)
        self.tb.add_scalars('Loss', {"train_loss": loss, "test_loss": test_loss}, self.epoch_count)
        self.tb.add_scalars('Accuracy', {"train_accuracy": accuracy, "test_accuracy": test_accuracy}, self.epoch_count)
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['loss'] = loss
        results['accuracy'] = accuracy
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run_params._asdict().items():
            results[k] = v
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')
        clear_output(wait = True)
        display(df)
    def get_num_workers(self, num_workers):
        self.epoch_num_workers = num_workers
    def track_loss(self, loss, batch):
        self.epoch_loss += loss.item() * batch[0].shape[0]
    def test_loss(self, test_loss, test_batch):
        self.test_epoch_loss += test_loss.item() * test_batch[0].shape[0]
    def test_num_correct(self, test_preds, test_labels):
        self.test_epoch_correct_num += self.get_correct_num(test_preds, test_labels)
    def track_num_correct(self, preds, labels):
        self.epoch_correct_num += self.get_correct_num(preds, labels)
    def get_correct_num(self, preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()
    def save(self, fileName):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{fileName}.csv')
        with open(f'{fileName}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run_data, f, ensure_ascii=False, indent=4)


class GTZANDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
    def __len__(self):
        return len(self.annotations)
    # Auto-called for each access, data standardization
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label, audio_sample_path
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal
    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
            signal = resampler(signal)
        return signal
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    def _get_audio_sample_path(self, index):
        fold = f"{self.annotations.iloc[index, -2]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[index, 1])
        return path
    def _get_audio_sample_label(self, index):
        return self.annotations.iloc[index, -1]

# =====================
# AlexNet model definition
# =====================
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(12288, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    ANNOTATIONS_FILE = "./features_30_sec_final.csv"
    AUDIO_DIR = "./GTZAN/genres_original"
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050 * 5
    plot = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=40,
        log_mels=True
    )
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    gtzan = GTZANDataset(
        ANNOTATIONS_FILE,
        AUDIO_DIR,
        mfcc,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )
    print(f"There are {len(gtzan)} samples in the dataset")

    if plot:
        signal, label, path = gtzan[666]
        print(f'path:{path}')
        signal = signal.cpu()
        print(signal.shape)
        plt.figure(figsize=(16, 8), facecolor="white")
        plt.imshow(signal[0,:,:], origin='lower')
        plt.autoscale(False)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.colorbar()
        plt.axis('auto')
        plt.show()

    print("\033[92mStart train...\033[0m")  # Green text for training start

    # Training related functions
    def create_data_loader(train_data, batch_size):
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        return train_dataloader
    def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
        for input, target in data_loader:
            input, target = input.to(device), target.to(device)
            prediction = model(input)
            loss = loss_fn(prediction, target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        print(f"loss: {loss.item()}")
    def train(model, data_loader, loss_fn, optimiser, device, epochs):
        for i in range(epochs):
            print(f"Epoch {i+1}")
            train_single_epoch(model, data_loader, loss_fn, optimiser, device)
            print("---------------------------")
        print("Finished training")

    # AlexNet structure display
    from torchsummary import summary
    alex = AlexNet().to(device)
    summary(alex, (1, 128, 111 * 5))
    torch.manual_seed(128)

    # Hyperparameter settings
    params = OrderedDict(
        lr=[.001, .0001],
        batch_size=[64],
        num_workers=[0],
        device=[device],
    )
    # Training and test set file paths
    ANNOTATIONS_FILE = "./features_30_sec_final.csv"
    AUDIO_DIR = "./GTZAN/genres_original"
    ANNOTATIONS_FILE_TEST = "./features_30_sec_test_final.csv"
    AUDIO_DIR_TEST = "./GTZAN_TEST/genres_original"

    mfcc = torchaudio.transforms.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=128,
        log_mels=True
    )
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    m = RunManager()
    for run in RunBuilder.get_runs(params):
        usd = GTZANDataset(ANNOTATIONS_FILE, AUDIO_DIR, mfcc, SAMPLE_RATE, NUM_SAMPLES, run.device)
        usd_test = GTZANDataset(ANNOTATIONS_FILE_TEST, AUDIO_DIR_TEST, mfcc, SAMPLE_RATE, NUM_SAMPLES, run.device)
        print(run)
        device = torch.device(run.device)
        train_data_loader = DataLoader(usd, batch_size=run.batch_size, num_workers=run.num_workers, shuffle=True)
        test_data_loader = DataLoader(usd_test, batch_size=run.batch_size, num_workers=run.num_workers)
        network = AlexNet().to(device)
        print(network)
        optimizer = optim.Adam(network.parameters(), lr=run.lr)
        m.begin_run(run, network, train_data_loader, test_data_loader)
        best_loss = float('inf')
        for epoch in range(100):
            # Training mode
            network.train()
            m.begin_epoch()
            for batch in train_data_loader:
                input = batch[0].to(device)
                target = batch[1].to(device)
                preds = network(input)
                loss = F.cross_entropy(preds, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                m.track_loss(loss, batch)
                m.track_num_correct(preds, target)
            with torch.no_grad():
                for test_batch in test_data_loader:
                    test_input = test_batch[0].to(device)
                    test_target = test_batch[1].to(device)
                    test_preds = network(test_input)
                    test_loss = F.cross_entropy(test_preds, test_target)
                    m.test_loss(test_loss, test_batch)
                    m.test_num_correct(test_preds, test_target)
            m.end_epoch()
            # Output detailed information for each epoch
            train_size = len(train_data_loader.dataset)
            test_size = len(test_data_loader.dataset)
            train_loss = m.epoch_loss
            train_acc = m.epoch_correct_num / train_size if train_size else 0
            test_loss = m.test_epoch_loss
            test_acc = m.test_epoch_correct_num / test_size if test_size else 0
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        torch.save(network.state_dict(), f'best_model_okk.pth')
        m.end_run()
        m.save(f'{run.lr}_{run.batch_size}')


class_mapping = [
    'blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'
]
