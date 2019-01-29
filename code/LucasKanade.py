import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0=np.zeros(2)):
    # Input:
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	p0: Initial movement vector [dp_x0, dp_y0]
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here


    # pad_width = int(It.shape[1]/2)
    # It = np.pad(It, pad_width, mode='symmetric')
    # It1 = np.pad(It1, pad_width, mode='symmetric')

    p = p0

    width_patch = np.round(rect[2]  - rect[0] +1)
    height_patch = np.round(rect[3] - rect[1] +1)

    gradX = cv2.Sobel(It1, -1, 1, 0, borderType=cv2.BORDER_CONSTANT)
    gradY = cv2.Sobel(It1, -1, 0, 1, borderType=cv2.BORDER_CONSTANT)
    # gradX, gradY = np.gradient(It1)

    ###################################  SPLINES  ########################################
    spline_It1 = RectBivariateSpline( np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    spline_It1_X_grad = RectBivariateSpline( np.arange(It1.shape[0]), np.arange(It1.shape[1]), gradX)
    spline_It1_Y_grad = RectBivariateSpline( np.arange(It1.shape[0]), np.arange(It1.shape[1]), gradY)

    threshold = 1.5e-3
    p_delta_norm = np.inf

    while(p_delta_norm>threshold):
        warp = rect + [ p[0], p[1], p[0], p[1] ]

        jacobian = np.asarray([ [1,0],
                                [0,1] ])

        ################################### INTERPOLATING IT WARP ##################################
        It1_patch = spline_It1.__call__(np.linspace(warp[1], warp[3], height_patch, endpoint=True),
                                        np.linspace(warp[0], warp[2], width_patch, endpoint=True))

        #####################################  TEMPLATE ERROR  #####################################
        b = It - It1_patch
        b_vect = b.reshape( 1, b.shape[0]*b.shape[1])
        # print('b flatten shape: ', b.shape)

        ################################### INTERPOLATING GRAD X and Y WARP ################################
        gradX_It1_patch = spline_It1_X_grad.__call__(np.linspace(warp[1], warp[3], height_patch, endpoint=True),
                                        np.linspace(warp[0], warp[2], width_patch, endpoint=True))
        gradY_It1_patch = spline_It1_Y_grad.__call__(np.linspace(warp[1], warp[3], height_patch, endpoint=True),
                                        np.linspace(warp[0], warp[2], width_patch, endpoint=True))

        gradX_It1_patch_vect =  gradX_It1_patch.reshape( gradX_It1_patch.shape[0]* gradX_It1_patch.shape[1]  ,1)
        gradY_It1_patch_vect =  gradY_It1_patch.reshape( gradY_It1_patch.shape[0]* gradY_It1_patch.shape[1]  ,1)

        ################################### STACKING GRAD X and Y WARP ################################
        grad_It1_patch = np.hstack( ( gradX_It1_patch_vect, gradY_It1_patch_vect) )
        # print('grad_It1_patch shape: ', grad_It1_patch.shape)

        A = np.matmul(grad_It1_patch , jacobian)
        # print('A: ', A)
        # print('A shape: ', A.shape)

        hessian = np.matmul(A.T, A)
        # print('Hessian shape: ', hessian.shape)

        p_delta = np.matmul( np.linalg.inv(hessian) ,   np.matmul(A.T, b_vect.T)   )
        p_delta = p_delta.flatten()
        # print('p_delta shape: ',p_delta.shape)

        p += p_delta

        p_delta_norm = np.sqrt(p_delta[0]**2 + p_delta[1]**2)
        print('image error: ', b.sum(), ', p_delta: ', p_delta, ', norm: ', p_delta_norm)

    return p
