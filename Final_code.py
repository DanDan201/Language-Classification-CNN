
#list of imports 
import numpy as np
import torch
import torch.nn as nn
from IPython.display import Audio
import matplotlib.pyplot as plt
import librosa
from torchsummary import summary
import torchaudio
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncode


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))



#dataloading
sampling_rate = 8_000
languages = ["de", "en", "es", "fr", "nl", "pt"]
language_dict = {languages[i]: i for i in range(len(languages))}

X_train, y_train = np.load("dataset/inputs_train_fp16.npy"), np.load(
    "dataset/targets_train_int8.npy"
)
X_test, y_test = np.load("dataset/inputs_test_fp16.npy"), np.load(
    "dataset/targets_test_int8.npy"
)

X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)


#define mean min max
mean = (np.mean(X_train))
min = np.min(X_train)
max = np.max(X_train)
peak = np.max(np.abs(X_train))
desired_peak_level = 0.5  # Specify the desired peak level in dBFS
scale = desired_peak_level/peak


#normalization scaled to min max 
class Normalization(nn.Module):
    def __init__(self,scale,min,max):
        super().__init__()
        self.register_buffer("min",torch.tensor(min))
        self.register_buffer("max", torch.tensor(max))
        self.register_buffer("scale", torch.tensor(scale))


    def forward(self,x):
        with torch.no_grad():
            x = x*self.scale
            x = (x-self.min)/(self.max-self.min)*2 -1
        return x
norm = Normalization( scale,min,max)


#transformation to Mel spectogram 
class TransformMel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=8000,
            n_fft=1024,
            hop_length=512,
            n_mels = 64
        )

    def forward(self,x):
        with torch.no_grad():
            x_mel = torch.stack([self.transform(i) for i in x])
            x_mel = x_mel.unsqueeze(1).float()
        return x_mel
transform = TransformMel()



#turn into tensor 
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.from_numpy(y_train)
y_test = torch.from_numpy(y_test)




#encode the class label 
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)




#Split training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)



#Define custom dataset

class MelSpectrogramDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

train_dataset = MelSpectrogramDataset(X_train, y_train)
val_dataset = MelSpectrogramDataset(X_val, y_val)
test_dataset = MelSpectrogramDataset(X_test, y_test)


# Put dataset in the dataloader


batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


# CNN model



class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.norm = norm
        self.transform = transform
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(32,64,kernel_size=3, stride = 1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = nn.Conv2d(64,128,kernel_size=3,stride =1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*30,64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(64,num_classes)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.transform(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        return x

num_classes = 6
model = CNNModel(num_classes)

# summary(model,(32,40000))


# Train the model

# In[277]:


num_epochs = 10
criterion = nn.CrossEntropyLoss()
#optimizer and learning rate is adjusted manually 
optimizer = optim.Adam(model.parameters(), lr=0.0005)
model = model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {train_loss/len(train_loader):.4f} - Validation Loss: {val_loss/len(val_loader):.4f}")


# Check the accuracy


model.eval()
correct = 0
total = 0
true_labels = []
predicted_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim=1)
        predicted_labels.extend(predicted.numpy())
        true_labels.extend(labels.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print('Test accuracy:', test_accuracy)

print(classification_report(true_labels, predicted_labels, target_names=languages))

cm = confusion_matrix(true_labels,predicted_labels)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=languages)
disp.plot()
plt.show()

