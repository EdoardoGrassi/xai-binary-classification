import torch as tc

def sliding_window(steps: tuple[int, int], stride: tuple[int, int]):
    for h in range(0, steps[0]):
        for w in range(0, steps[1]):
            yield h * stride[0], w * stride[1]

def generate_occlusion_masks(
    device: tc.device,
    mask_shape: tuple[int, int],
    window_shape: tuple[int, int],
    window_stride: tuple[int, int],
) -> tc.Tensor:
    # assume image has standard pytorch format [B, C, H, W]
    kh, kw = window_shape
    sh, sw = window_stride
    # number of applicable windows along each image dimension
    nh: int = (mask_shape[0] - kh + sh) // sh
    nw: int = (mask_shape[1] - kw + sw) // sw

    # generate a batch of images
    masks = tc.zeros((nh * nw, *mask_shape), dtype=tc.bool, device=device)
    # occlude with rolling window
    for i, (h, w) in enumerate(sliding_window((nh, nw), window_stride)):
        masks[i, h : h + kh, w : w + kw] = True

    return masks

def generate_occluded_batch(
    image: tc.Tensor,
    window_shape: tuple[int, int],
    window_stride: tuple[int, int],
    baseline: float,
) -> tuple[tc.Tensor, tc.Tensor]:
    assert image.ndim == 3, "Expected a single image"
    # TODO: add support for batches

    # generate a batch of images
    h, w = image.shape[-2:]
    masks = generate_occlusion_masks(image.device, (h, w), window_shape, window_stride)
    batch = image.unsqueeze(0).repeat(masks.shape[0], 1, 1, 1)
    # occlude with rolling window
    # assume image has standard pytorch format [B, C, H, W]
    # batch[masks[:, tc.newaxis, :, :]] = baseline
    batch.masked_fill_(masks[:, tc.newaxis, :, :], baseline)
    return masks, batch

def generate_occluded_batches(
    images: tc.Tensor,
    window_shape: tuple[int, int],
    window_stride: tuple[int, int],
    baseline: float,
) -> tuple[tc.Tensor, tc.Tensor]:
    assert images.ndim == 4, "Expected a batch of images"
    # assume image has standard pytorch format [B, C, H, W]

    # generate a batch of images
    h, w = images.shape[-2:]
    masks = generate_occlusion_masks(images.device, (h, w), window_shape, window_stride)

    # change to [B, mask, C, H, W]
    batch = images.unsqueeze(dim=1).repeat(1, masks.shape[0], 1, 1, 1)
    # occlude with rolling window
    batch.masked_fill_(masks[tc.newaxis, :, tc.newaxis, :, :], baseline)
    return masks, batch


class ImageOcclusion:
    def __init__(self) -> None:
        pass

    def explain_image(
        self,
        model: tc.nn.Module,
        image: tc.Tensor,
        window_shape: tuple[int, int],
        window_stride: tuple[int, int],
        baseline: float,
    ) -> tc.Tensor:
        assert image.ndim == 3
        masks, perturbations = generate_occluded_batch(image, window_shape, window_stride, baseline)
        with tc.inference_mode():
            # some models always require a batch dimension so add one
            reference = model(image.view(1, *image.shape)).squeeze(0)
            predictions = model(perturbations)
            assert predictions.shape[-1] == 1
            reference.squeeze_(-1)
            predictions.squeeze_(-1)

        importance = reference - predictions
        # stack the scores into a heatmap
        heatmap = importance[:, tc.newaxis, tc.newaxis] * masks.float()
        heatmap = tc.sum(heatmap, dim=0)

        # normalize each pixel score based on number of occlusion masks that cover it
        mask_cover_count = masks.int().sum(dim=0)
        heatmap /= mask_cover_count
        return heatmap
        
    def explain_batch(
        self,
        model: tc.nn.Module,
        images: tc.Tensor,
        window_shape: tuple[int, int],
        window_stride: tuple[int, int],
        baseline: float,
    ) -> tc.Tensor:
        assert images.ndim == 4
        batch_size: int = images.shape[0] 
        masks, perturbations = generate_occluded_batches(images, window_shape, window_stride, baseline)
        with tc.inference_mode():
            reference = model(images)
            # merge batch and mask dimension for model prediction
            predictions = model(perturbations.view(-1, *images.shape[-3:])).view(batch_size, -1, 1)
            assert predictions.shape[-1] == 1
            reference.squeeze_(-1)
            predictions.squeeze_(-1)

        importance = reference - predictions
        # stack the scores into a heatmap
        heatmap = importance[:, tc.newaxis, tc.newaxis] * masks.float()
        heatmap = tc.sum(heatmap, dim=1)

        # normalize each pixel score based on number of occlusion masks that cover it
        mask_cover_count = masks.int().sum(dim=0)
        heatmap /= mask_cover_count
        return heatmap
        