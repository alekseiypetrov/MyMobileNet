import os.path
import coremltools as ct
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_set = torchvision.datasets.ImageFolder(root='./ProcessedDataset/train', transform=train_transform)
val_set = torchvision.datasets.ImageFolder(root='./ProcessedDataset/val', transform=val_test_transform)
test_set = torchvision.datasets.ImageFolder(root='./ProcessedDataset/test', transform=val_test_transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=6)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=6)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=6)


class MyMobileNet(nn.Module):
    # Загрузка и настройка предобученной модели MobileNetV3
    def __init__(self, classes):
        super().__init__()
        self.model = models.mobilenet_v3_large(weights="DEFAULT")
        self.model_classes = classes

        for param in self.model.features.parameters():
            param.requires_grad = False
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, len(self.model_classes))

        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.best_val_acc = 0.0

        # Функция потерь и оптимизатор
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)

    # Обучение модели
    def train_model(self, trainloader, valloader, epochs=10):
        patience = 3
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in trainloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = (running_loss / len(trainloader))
            train_acc = 100 * correct / total

            val_loss, val_acc = self.evaluate(valloader)

            print(f"\nEpoch {epoch + 1}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Accuracy: {train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.2f}%")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), "best_mobilenetv3.pth")
                print("Лучшая модель сохранена.")
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print("\nСработала ранняя остановка.")
                break

        print('Обучение завершено')

    # Вычисление потери и точности
    def evaluate(self, dataloader):
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = (running_loss / len(dataloader))
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    # Расчет метрик для итогового отчета
    def test_model(self, testloader):
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.numpy())
                y_pred.extend(predicted.cpu().numpy())

        print("\nРЕЗУЛЬТАТЫ ПРОВЕРКИ")
        print(classification_report(y_true, y_pred, target_names=self.model_classes))
        print("\nМАТРИЦА ОШИБОК")
        print(confusion_matrix(y_true, y_pred))

    # конвертация для CoreML
    def convert_to_coreml(self):
        self.model.eval()
        example_input = torch.rand(1, 3, 224, 224).to(self.device)
        traced_model = torch.jit.trace(self.model, example_input)
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.ImageType(
                name="input_image",
                shape=example_input.shape,
                scale=1 / 255.0 * 0.229,
                bias=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])
            ],
            classifier_config=ct.ClassifierConfig(self.model_classes)
        )

        coreml_model.save("GestureClassifier.mlpackage")
        print("\nМодель CoreML сохранена:")
        print("GestureClassifier.mlpackage")


if __name__ == '__main__':
    TRAIN_MODEL = False
    model = MyMobileNet(train_set.classes)
    if TRAIN_MODEL or not os.path.exists("./best_mobilenetv3.pth"):
        model.train_model(train_loader, val_loader, epochs=10)

    model.model.load_state_dict(torch.load("best_mobilenetv3.pth", weights_only=True))
    model.test_model(test_loader)

    # выполнение конвертации осуществляется исключительно на MacOS
    model.convert_to_coreml()
