import numpy as np
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt


def InverseCompositionAffine(It, It1):
    # Input:
    #	It: template image
    #	It1: Current image
    # Output:
    #	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0, 0, 1]])

    ###############################  PRE COMPUTATION  ####################################
    gradY, gradX = np.gradient(It)

    gradX_It_vect = gradX.flatten()
    gradY_It_vect = gradY.flatten()

    #######################  RESULT AFTER MULTIPLICATION WITH JACOBIAN  ######################
    A = np.zeros((gradX_It_vect.shape[0], 6))
    rows, cols = np.where(It != -1)

    A[:, 0] = np.multiply(gradX_It_vect, cols)
    A[:, 1] = np.multiply(gradX_It_vect, rows)
    A[:, 2] = gradX_It_vect
    A[:, 3] = np.multiply(gradY_It_vect, cols)
    A[:, 4] = np.multiply(gradY_It_vect, rows)
    A[:, 5] = gradY_It_vect
    # print('A shape: ', A.shape, ' , A: ', A)
    # plt.imshow( A[:, 2].reshape( (240,320) ) )
    # plt.show()
    # plt.waitforbuttonpress()

    hessian = np.matmul(A.T, A)
    # print('Hessian shape: ', hessian.shape)

    threshold = 5e-3

    p_delta = np.zeros(6)
    p_delta_norm = np.inf

    while (p_delta_norm > threshold):

        M_delta = np.array([ [ 1 + p_delta[4], p_delta[3], p_delta[5] ],
                       [ p_delta[1], 1 + p_delta[0], p_delta[2] ],
                       [0, 0, 1] ])

        M = np.matmul( M, np.linalg.inv(M_delta) )
        # print('M: ', M)

        It1_warp = affine_transform(It1, M)
        mask = np.ones( It1.shape )
        mask  = affine_transform(mask, M)
        # print(mask)

        # plt.imshow( It1_warp)
        # plt.show()
        # plt.waitforbuttonpress()

        It_patch = np.multiply( It, mask )

        b = It1_warp - It_patch
        b_vect = b.reshape( b.shape[0]*b.shape[1], 1 )
        # print('b_vect shape : ', b_vect.shape)

        p_delta = np.matmul(np.linalg.inv(hessian), np.matmul(A.T, b_vect))
        p_delta = p_delta.flatten()
        # print('p_delta: ', p_delta)
        # print('p_delta shape: ',p_delta.shape)

        p_delta_norm = np.linalg.norm(p_delta, 2)
        # print('p: ', p, ' , norm: ', p_delta_norm)
        print('image error: ', b.sum(), ', p_delta: ', p_delta, ', norm: ', p_delta_norm)

    return M
