import torch

def train(model, num_epochs, train_loader, optim, criterion, device):
    for epoch in range(num_epochs):
        model.train()

        training_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optim.zero_grad()

            output = model(X_batch)

            loss = criterion(output, y_batch)
            loss.backward()
            optim.step()

            training_loss += loss.item()

        print(f"Epoch: {epoch+1}/{num_epochs}, Training loss: {training_loss/len(train_loader)}")
        torch.save(model.state_dict(), "iris_model.pth")