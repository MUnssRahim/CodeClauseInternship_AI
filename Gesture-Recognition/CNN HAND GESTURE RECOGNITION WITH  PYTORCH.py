import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 16, 100)
        self.fc2 = nn.Linear(100, 75)
        self.fc3 = nn.Linear(75, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 5 * 5 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def trainandsavemodel():
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainData = ImageFolder(
        root='C:\\Users\\HP\\Desktop\\images',
        transform=transform
    )

    testData = ImageFolder(
        root='C:\\Users\\HP\\Desktop\\images',
        transform=transform
    )

    trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
    testLoader = DataLoader(testData, batch_size=64, shuffle=False)

    epochs = 10
    trainLosses = []
    testLosses = []

    for epoch in range(epochs):
        model.train()
        runningLoss = 0.0  
        for i, (XTrain, yTrain) in enumerate(trainLoader):
            optimizer.zero_grad()
            yPred = model(XTrain)
            loss = criterion(yPred, yTrain)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}], Loss: {runningLoss / 100}')
                runningLoss = 0.0

        model.eval()
        testLoss = 0.0  
        correct = 0  
        
        with torch.no_grad():
            for XTest, yTest in testLoader:
                yPred = model(XTest)
                loss = criterion(yPred, yTest)
                testLoss += loss.item()

                predictedClasses = torch.argmax(yPred, dim=1)
                correctPredictions = (predictedClasses == yTest).sum().item()
                correct += correctPredictions

        avgTestLoss = testLoss / len(testLoader)
        accuracy = 100 * correct / len(testData)
        testLosses.append(avgTestLoss)

        print(f'Epoch [{epoch + 1}/{epochs}], Test Loss: {avgTestLoss:.4f}, Accuracy: {accuracy:.2f}%')

    print('Training complete')

    model_path = 'CNNmodel.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model saved as {model_path}')

def load_model(model_path):
    model = SimpleCNN()
    try:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print(f'Model loaded from {model_path}')
        else:
            raise FileNotFoundError(f"Model file '{model_path}' does not exist.")
    except FileNotFoundError as e:
        print(e)
        exit(1)
    except RuntimeError as e:
        print(f"Error loading the model: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)
    return model

def classify_image(image_path, model, class_names):
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label_index = predicted.item()
        label_name = class_names[label_index]
        print(f'Predicted Label: {label_name}')

def main():
    model_path = 'CNNmodel.pth'
    model = load_model(model_path)
    
    class_names = ['Closed Fist', 'Finger Circle', 'Finger symbol', 'Finger Bend', 'OpenPalm', 'SemiOpenFist', 'Semi OpenPalm', 'Single FingerBend']

    image_path = r'C:\\Users\\HP\\Desktop\\img1.jpg'

    if os.path.isfile(image_path):
        classify_image(image_path, model, class_names)
    else:
        print("Error: File does not exist.")

main()
trainandsavemodel()
