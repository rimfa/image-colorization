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


def train_model(model, criterion, optimizer, train_loader, val_loader,
                num_epochs, device, last_path=None, best_path=None,
                run=None, scheduler=None, resume_from=None):

    # Инициализация
    train_loss = []
    val_loss = []
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device, weights_only=True)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        train_loss = checkpoint["train_losses"].copy()
        val_loss = checkpoint["val_losses"].copy()
        best_val_loss = min(val_loss)
        start_epoch = checkpoint["epoch"]
    else:
        best_val_loss = float("inf")
        start_epoch = 0

    # Цикл обучения по эпохам
    pbar = tqdm(range(start_epoch + 1, start_epoch + num_epochs + 1), desc="Training")

    for epoch in pbar:

        # Обучение и валидация
        train_loss.append(training_epoch(
            model, optimizer, criterion, train_loader, device
        ))

        val_loss.append(validation_epoch(
            model, criterion, val_loader, device
        ))

        # Обновление learning rate
        if scheduler is not None:
            scheduler.step(val_loss[-1])

        # Формирование checkpoint
        if (last_path is not None) or (best_path is not None):
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": (
                    scheduler.state_dict() if scheduler is not None else None
                ),
                "train_losses": train_loss,
                "val_losses": val_loss
            }

            # Сохранение последнего состояния модели
            if last_path is not None:
                torch.save(checkpoint, last_path)

            # Сохранение лучшей модели по validation loss
            if best_path is not None and val_loss[-1] < best_val_loss:
                best_val_loss = val_loss[-1]
                torch.save(checkpoint, best_path)

        # Логирование метрик в W&B
        learning_rate = optimizer.param_groups[0]["lr"]

        if run is not None:
            run.log({
                "epoch": epoch,
                "train_loss": train_loss[-1],
                "val_loss": val_loss[-1],
                "learning_rate": learning_rate
            })

        # Обновление прогресса обучения в tqdm
        pbar.set_postfix(
            train_loss=f"{train_loss[-1]:.4f}",
            val_loss=f"{val_loss[-1]:.4f}",
            lr=f"{learning_rate:.2e}"
        )

    # Возврат истории обучения
    return train_loss, val_loss