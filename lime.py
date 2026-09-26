import numpy as np
import torch as tc
from sklearn.linear_model import LinearRegression
from skimage.segmentation import felzenszwalb
from scipy.spatial.distance import cdist

def radial_basis_kernel(distances: np.ndarray, width: float):
    return np.sqrt(np.exp(-(distances ** 2) / width ** 2))

class BinaryImageLime:
    """LIME explanations for a binary classifier that operates on image data."""

    def __init__(self, rbf_kernel_width: float) -> None:
        assert rbf_kernel_width > 0

        self.rbf_kernel_width = rbf_kernel_width


    def explain(self, model: tc.Module, target: tc.Tensor, n_samples: int) -> tc.Tensor:
        """
        model: Block-box model
        target: Image sample for which to produce explanations
        """
        assert n_samples >= 0

        # generate perturbated samples
        sampled_data = self._generate_sample_variations(target.numpy())

        distances = cdist(target, sampled_data, metric="euclidean")
        sampled_data_weights = radial_basis_kernel(distances, width=self.rbf_kernel_width)

        # extract predictions from the original model
        sampled_data_labels = model(sampled_data)

        # fit surrogate model
        surrogate_local_model = LinearRegression(fit_intercept=True)
        surrogate_local_model.fit(sampled_data, sampled_data_labels, sampled_data_weights)

        # export weights mask
        pass

    def _generate_sample_variations(self, model: tc.Module, sample: np.ndarray, n_samples: int):
        sampled_data = np.stack([])
        image_patch_indices = felzenszwalb(sample)
        # grey color filler
        filler = np.array((0.5, 0.5, 0.5))

        # convert indices to boolean masks
        n_patches = np.unique(image_patch_indices).shape[0]
        image_patch_masks = np.stack([image_patch_indices == i for i in range(n_patches)])

        # random selection of masks for each image
        data = self.random_state.randint(0, 2, (n_samples, n_patches))
        data[0, :] = 1

        imgages = sample + image_patch_masks * filler

        for row in data:
            temp = copy.deepcopy(image)
            zeros = np.where(row == 0)[0]
            mask = np.zeros(segments.shape).astype(bool)
            for z in zeros:
                mask[segments == z] = True
            temp[mask] = fudged_image[mask]
            imgs.append(temp)

        labels = model(imgs)
        return data, np.array(labels)
