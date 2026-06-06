from torchmetrics.image.kid import KernelInceptionDistance


def evaluate(real_images, generated_images, device, batch_size=16):
    subset_size = min(50, len(real_images), len(generated_images))

    kid = KernelInceptionDistance(
        subset_size=subset_size,
        normalize=True,
    ).to(device)

    for batch in real_images.split(batch_size):
        kid.update(batch.to(device), real=True)

    for batch in generated_images.split(batch_size):
        kid.update(batch.to(device), real=False)

    mean, std = kid.compute()
    return {
        "mean": mean.item(),
        "std": std.item(),
    }