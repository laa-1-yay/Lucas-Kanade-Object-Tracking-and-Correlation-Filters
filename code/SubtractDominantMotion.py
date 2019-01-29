import numpy as np
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import affine_transform, binary_erosion, binary_dilation, binary_closing


def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1
	# Output:
	#	mask: [nxm]
	# put your implementation here

	mask = np.ones(image1.shape, dtype=bool)

	# M = LucasKanadeAffine(image1, image2)

	M = InverseCompositionAffine(image1, image2)

	It_warp = affine_transform(image1, np.linalg.inv(M))

	warped_black_pixels_mask = np.ones(image1.shape)
	warped_black_pixels_mask = affine_transform(warped_black_pixels_mask, np.linalg.inv(M))
	image2_common = np.multiply(image2, warped_black_pixels_mask)

	diff = np.abs(image2_common - It_warp)
	non_moving_pixels = np.where(diff <= 0.13)
	mask[non_moving_pixels] = False

	mask = binary_dilation(mask)
	mask = binary_erosion(mask)
	mask = binary_closing(mask, iterations=9)
	mask = binary_dilation(mask)

	return mask
