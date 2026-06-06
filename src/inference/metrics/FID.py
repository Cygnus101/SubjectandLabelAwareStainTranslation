from torchmetrics.image.fid import FrechetInceptionDistance


def evaluate(real_images, generated_images, device, batch_size=16):
    fid = FrechetInceptionDistance(normalize=True).to(device)

    for batch in real_images.split(batch_size):
        fid.update(batch.to(device), real=True)

    for batch in generated_images.split(batch_size):
        fid.update(batch.to(device), real=False)

    return fid.compute().item()