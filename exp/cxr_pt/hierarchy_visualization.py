from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


def denormalize_image(pixel_values, image_mean, image_std):
    """Convert one normalized CHW processor tensor to an RGB PIL image."""
    image = pixel_values.detach().float().cpu()
    mean = torch.as_tensor(image_mean, dtype=image.dtype)[:, None, None]
    std = torch.as_tensor(image_std, dtype=image.dtype)[:, None, None]
    image = (image * std + mean).clamp(0, 1)
    image = (image.permute(1, 2, 0) * 255).round().byte().numpy()
    return Image.fromarray(image, mode="RGB")


def save_anatomy_mask_contact_sheet(
    image,
    masks,
    labels,
    valid_mask,
    output_path,
    threshold=0.5,
    columns=6,
):
    """Save all anatomy masks overlaid on the exact processed model image."""
    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL image")
    if masks.ndim == 2:
        side = int(masks.shape[-1] ** 0.5)
        if side * side != masks.shape[-1]:
            raise ValueError("flattened masks must contain a square token grid")
        masks = masks.reshape(masks.shape[0], side, side)
    if masks.ndim != 3 or len(labels) != masks.shape[0]:
        raise ValueError("masks must have shape [A, H, W] matching labels")
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool)
    if valid_mask.shape != (masks.shape[0],):
        raise ValueError("valid_mask must have shape [A]")

    tile_width, tile_height = image.size
    header_height = 28
    rows = (len(labels) + columns - 1) // columns
    sheet = Image.new(
        "RGB", (columns * tile_width, rows * (tile_height + header_height)), "white"
    )
    font = ImageFont.load_default()
    resized = F.interpolate(
        masks[:, None].detach().float().cpu(),
        size=(tile_height, tile_width),
        mode="nearest",
    )[:, 0]
    for index, label in enumerate(labels):
        row, column = divmod(index, columns)
        x = column * tile_width
        y = row * (tile_height + header_height)
        tile = image.copy()
        if valid_mask[index]:
            mask = resized[index] >= threshold
            red = Image.new("RGB", image.size, (255, 32, 32))
            alpha = Image.fromarray((mask.numpy() * 105).astype("uint8"), mode="L")
            tile = Image.composite(red, tile, alpha)
        sheet.paste(tile, (x, y + header_height))
        draw = ImageDraw.Draw(sheet)
        status = "GT" if valid_mask[index] else "UNANNOTATED"
        draw.text((x + 4, y + 7), f"{index:02d} {label} [{status}]", fill="black", font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return output_path
