import numpy as np
from scipy.ndimage import affine_transform

def LucasKanadeAffine(It, It1):
    # Input:
    #	It: template image
    #	It1: Current image
    # Output:
    #	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1]])

    p = np.zeros(6)

    # gradX = cv2.Sobel(It1, -1, 1, 0, borderType=cv2.BORDER_CONSTANT)
    # gradY = cv2.Sobel(It1, -1, 0, 1, borderType=cv2.BORDER_CONSTANT)
    gradY, gradX = np.gradient(It1)

    threshold = 5e-3
    p_delta_norm = np.inf

    while (p_delta_norm > threshold):

        M = np.array([ [ 1 + p[4], p[3], p[5] ],
                       [ p[1], 1 + p[0], p[2] ],
                       [0, 0, 1] ])

        It1_warp = affine_transform(It1, M)
        mask = np.ones( It1.shape )
        mask  = affine_transform(mask, M)

        # plt.imshow( It1_warp)
        # plt.show()
        # plt.waitforbuttonpress()

        It_patch = np.multiply( It, mask )

        rows, cols = np.where(It1_warp != -1)

        b = It_patch - It1_warp
        b_vect = b.reshape( It_patch.shape[0]*It_patch.shape[1], 1 )
        # print('b flatten shape: ', b_vect.shape)

        gradX_patch = affine_transform(gradX, M)
        gradY_patch = affine_transform(gradY, M)

        gradX_It1_patch_vect = gradX_patch.flatten()
        gradY_It1_patch_vect = gradY_patch.flatten()

        #######################  RESULT AFTER MULTIPLICATION WITH JACOBIAN  ######################
        A = np.zeros(( gradX_It1_patch_vect.shape[0] ,6))

        A[:,0] = np.multiply(gradX_It1_patch_vect, cols)
        A[:,1] = np.multiply(gradX_It1_patch_vect, rows)
        A[:,2] = gradX_It1_patch_vect
        A[:,3] = np.multiply(gradY_It1_patch_vect, cols)
        A[:,4] = np.multiply(gradY_It1_patch_vect, rows)
        A[:,5] = gradY_It1_patch_vect
        # print('A shape: ', A.shape)
        # print('A: ', A)
        # plt.imshow( A[:, 2].reshape( (240,320) ) )
        # plt.show()
        # plt.waitforbuttonpress()

        hessian = np.matmul(A.T, A)
        # print('Hessian: ', hessian)
        # print('Hessian shape: ', hessian.shape)

        p_delta = np.matmul(np.linalg.inv(hessian), np.matmul(A.T, b_vect))
        p_delta = p_delta.flatten()
        # print('p_delta shape: ',p_delta.shape)

        p += p_delta

        p_delta_norm = np.linalg.norm(p_delta, 2)
        # print('p_delta: ', p_delta)
        # print('p: ', p, ' , norm: ', p_delta_norm)
        print('image error: ', b.sum(), ', p_delta: ', p_delta, ', norm: ', p_delta_norm)

    return M
