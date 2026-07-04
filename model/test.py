import torch

def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        correct, total = 0, 0

        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            output = model(X_batch)

            _, predicted = torch.max(output, dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

            accuracy = 100*(correct/total)

        print(f"Test accuracy: {accuracy:.2f}%")
