class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def train(model, device, train_loader, optimizer, epoch, log_step):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = t.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_step == 0:
            print(
                "train epoch : {}[{}/{} ({:.0f}%)]\tloss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with t.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += t.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        "\ntest set : average loss : {:.4f}, accuracy:{}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss


def main():
    train_loader = t.utils.data.DataLoader(
        dataset=dataset1,
        pin_memory=True,
        num_workers=1,
        shuffle=True,
        batch_size=4,
    )

    test_loader = t.utils.data.DataLoader(
        dataset=dataset2,
        pin_memory=True,
        num_workers=1,
        shuffle=False,
        batch_size=4,
    )

    epoch = 1000
    device = "cuda"
    model = Model().to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.7)
    early_stopper = EarlyStopper(patience=2, min_delta=0)

    for epoch in range(1, epoch + 1):
        train(
            device=device,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            log_step=100,
        )

        val_loss = test(model, device, test_loader)
        stop = early_stopper.early_stop(val_loss)
        if stop:
            print("epoch", epoch)
            break

        scheduler.step()


main()
