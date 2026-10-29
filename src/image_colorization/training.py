from tqdm.auto import tqdm
import torch

def training_epoch(model, optimizer, criterion, train_loader, device):
    total_loss = 0.0
    total_samples = 0

    model.train()
    for L, ab in train_loader:
        L = L.to(device)
        ab = ab.to(device)

        optimizer.zero_grad()
        pred = model(L)
        loss = criterion(pred, ab)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * L.size(0)
        total_samples += L.size(0)

    return total_loss / total_samples


def validation_epoch(model, criterion, val_loader, device):
    total_loss = 0.0
    total_samples = 0
    model.eval()

    with torch.no_grad():
        for L, ab in val_loader:
            L = L.to(device)
            ab = ab.to(device)

            pred = model(L)
            loss = criterion(pred, ab)
            total_loss += loss.item() * L.size(0)
            total_samples += L.size(0)

    return total_loss / total_samples


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, device,
          last_path=None, best_path=None, run=None):

    # Инициализация
    train_loss = []
    val_loss = []
    best_val_loss = float("inf")

    # Цикл обучения по эпохам
    pbar = tqdm(range(1, num_epochs + 1), desc="Training")

    for epoch in pbar:

        # Обучение и валидация
        train_loss.append(training_epoch(model, optimizer,
                                         criterion, train_loader, device))

        val_loss.append(validation_epoch(model, criterion,
                                         val_loader, device))
        if (last_path is not None) or (best_path is not None):
            # Формирование checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_losses": train_loss,
                "val_losses": val_loss
            }

            #Сохранение последнего состояния модели
            torch.save(checkpoint, last_path)

            # Сохранение лучшей модели по validation loss
            if val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                torch.save(checkpoint, best_path)

        # Логирование метрик в W&B
        if run is not None:
            run.log({
                "epoch": epoch,
                "train_loss": train_loss[-1],
                "val_loss": val_loss[-1]
            })

        # Обновление прогресса обучения в tqdm
        pbar.set_postfix(
            train_loss=f"{train_loss[-1]:.4f}",
            val_loss=f"{val_loss[-1]:.4f}"
        )

    #Возврат истории обучения
    return train_loss, val_loss