import sys
import os
import torch
from torch.utils.data import DataLoader

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from mvdot.data.preprocessing import load_mnist, create_noisy_view
from mvdot.data.dataset import noisy_mnist
from mvdot.models.autoencoder import autoencoder
from mvdot.losses.reconstruction import reconstruction_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images, labels = load_mnist()

noisy_images = create_noisy_view(
    images,
    sigma=0.25
)

data = noisy_mnist(
    images,
    noisy_images,
    labels,
    aligned_rate=1.0
)

loader = DataLoader(
    data,
    batch_size=512,
    shuffle=True,
    pin_memory=True
)

model1 = autoencoder().to(device)
model2 = autoencoder().to(device)

criterion = reconstruction_loss()

optimizer = torch.optim.Adam(
    list(model1.parameters()) + list(model2.parameters()),
    lr=0.0003
)

epochs = 10

for epoch in range(epochs):
    model1.train()
    model2.train()

    total_loss = 0.0

    for view1, view2, _ in loader:
        view1 = view1.to(device, non_blocking=True)
        view2 = view2.to(device, non_blocking=True)

        optimizer.zero_grad()

        output1, _ = model1(view1)
        output2, _ = model2(view2)

        loss1 = criterion(view1, output1)
        loss2 = criterion(view2, output2)

        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"epoch {epoch + 1}/{epochs} "
        f"loss: {average_loss:.6f}"
    )

torch.save(model1.state_dict(), "mvdot/models/autoencoder_view1.pt")
torch.save(model2.state_dict(), "mvdot/models/autoencoder_view2.pt")