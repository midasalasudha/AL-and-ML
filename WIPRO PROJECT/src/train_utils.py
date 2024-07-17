import torch
from sklearn.metrics import precision_score, recall_score, f1_score
def train_model(model, train_dl, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    train_loss = train_loss / len(train_dl.dataset)
    train_accuracy = correct_train / total_train

    return train_loss, train_accuracy

def validate_model(model, valid_dl, criterion, device):
    model.eval()
    valid_loss = 0.0
    correct_valid = 0
    total_valid = 0

    with torch.no_grad():
        for inputs, labels in valid_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * inputs.size(0)
            _, predicted_valid = torch.max(outputs.data, 1)
            total_valid += labels.size(0)
            correct_valid += (predicted_valid == labels).sum().item()

    valid_loss = valid_loss / len(valid_dl.dataset)
    valid_accuracy = correct_valid / total_valid

    return valid_loss, valid_accuracy

# def validate_model(model, valid_dl, criterion, device):
#     model.eval()
#     valid_loss = 0.0
#     correct_valid = 0
#     total_valid = 0
#     all_predictions = []
#     all_labels = []
#
#     with torch.no_grad():
#         for inputs, labels in valid_dl:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             valid_loss += loss.item() * inputs.size(0)
#
#             _, predicted_valid = torch.max(outputs.data, 1)
#             all_predictions.extend(predicted_valid.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())
#
#             total_valid += labels.size(0)
#             correct_valid += (predicted_valid == labels).sum().item()
#
#     valid_loss = valid_loss / len(valid_dl.dataset)
#     valid_accuracy = correct_valid / total_valid
#
#     # Calculate precision, recall and F1 score
#     precision = precision_score(all_labels, all_predictions, average='weighted')
#     recall = recall_score(all_labels, all_predictions, average='weighted')
#     f1 = f1_score(all_labels, all_predictions, average='weighted')
#
#     # print(f'Validation Loss: {valid_loss:.4f}, Validation Acc: {valid_accuracy:.4f}')
#     print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
#
#     return valid_loss, valid_accuracy, precision, recall, f1
