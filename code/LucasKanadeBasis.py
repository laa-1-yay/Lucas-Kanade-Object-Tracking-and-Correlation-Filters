import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeBasis(It, It1, rect, bases, p0=np.zeros(2)):
    # Input:
    #	It: template image
    #	It1: Current image
    #	rect: Current position of the car
    #	(top left, bot right coordinates)
    #	bases: [n, m, k] where nxm is the size of the template.
    # Output:
    #	p: movement vector [dp_x, dp_y]

    # Put your implementation here

    p = p0

    width_patch = np.round(rect[2]  - rect[0] +1)
    height_patch = np.round(rect[3] - rect[1] +1)

    # gradX = cv2.Sobel(It1, -1, 1, 0, borderType=cv2.BORDER_CONSTANT)
    # gradY = cv2.Sobel(It1, -1, 0, 1, borderType=cv2.BORDER_CONSTANT)
    gradY, gradX = np.gradient(It1)

    ###################################  SPLINES  ########################################
    spline_It1 = RectBivariateSpline( np.arange(It1.shape[0]), np.arange(It1.shape[1]), It1)
    spline_It1_X_grad = RectBivariateSpline( np.arange(It1.shape[0]), np.arange(It1.shape[1]), gradX)
    spline_It1_Y_grad = RectBivariateSpline( np.arange(It1.shape[0]), np.arange(It1.shape[1]), gradY)

    threshold = 1e-1
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
        # print('b flatten shape: ', b.shape)

        b_vect = b.reshape( b.shape[0]*b.shape[1], 1 )


        ################################### INTERPOLATING GRAD X and Y WARP ################################
        gradX_It1_patch = spline_It1_X_grad.__call__(np.linspace(warp[1], warp[3], height_patch, endpoint=True),
                                        np.linspace(warp[0], warp[2], width_patch, endpoint=True))
        gradY_It1_patch = spline_It1_Y_grad.__call__(np.linspace(warp[1], warp[3], height_patch, endpoint=True),
                                        np.linspace(warp[0], warp[2], width_patch, endpoint=True))

        gradX_It1_patch_vect =  gradX_It1_patch.reshape( gradX_It1_patch.shape[0]* gradX_It1_patch.shape[1]  ,1)
        gradY_It1_patch_vect =  gradY_It1_patch.reshape( gradY_It1_patch.shape[0]* gradY_It1_patch.shape[1]  ,1)

        ################################## CALCULATING B, WEIGHTS, BW ################################
        B = np.zeros( ( bases.shape[0]*bases.shape[1] , bases.shape[2]) )
        for i in range(bases.shape[2]):
            B[:,i] = bases[:,:,i].flatten()

        # error_vect = It1_patch - It
        # weights2 = B.T.dot(error_vect.reshape( error_vect.shape[0]*error_vect.shape[1] , 1))
        # print('weights1 : ', weights1)

        # weights = []
        # Bw_vect = np.zeros(   It.shape[0]* It.shape[1] )
        # for i in range(bases.shape[2]):
        #     base_vect = bases[:, :, i].reshape( bases.shape[0] * bases.shape[1])
        #     weight = np.dot(base_vect, -b_vect)
        #     weights.append(weight)
        #     Bw_vect += weight * base_vect
        # print('weights2 : ', weights2)

        ################################### STACKING GRAD X and Y WARP ################################
        grad_It1_patch = np.hstack( ( gradX_It1_patch_vect, gradY_It1_patch_vect ) )
        # print('grad_It1_patch shape: ', grad_It1_patch.shape)

        A = np.matmul(grad_It1_patch , jacobian)
        # print('A: ', A)
        # print('A shape: ', A.shape)

        ################################ UPDATION FOR APPEARANCE CHANGES #############################

        # Bw_vect = np.expand_dims(Bw_vect, 0)
        # print('Bw: ', Bw_vect)
        # print('Bw shape: ', Bw_vect.shape)

        # b_vect = b_vect + Bw_vect

        # gradX_It1_patch_vect = gradX_It1_patch_vect - Bw_vect.T
        # gradY_It1_patch_vect = gradY_It1_patch_vect - Bw_vect.T

        # print('b_vect shape: ', b_vect.shape)
        # print('gradX_It1_patch_vect shape: ', gradX_It1_patch_vect.shape)

        span_term = np.eye(B.shape[0], B.shape[0]) - np.matmul(B,B.T)
        # print('span_term: ', span_term)
        # print('span_term shape: ', span_term.shape)

        b_vect_new = np.matmul(span_term, b_vect)

        A_new = np.matmul(span_term, A)

        hessian = np.matmul(A_new.T, A_new)
        # print('Hessian shape: ', hessian.shape)

        p_delta = np.matmul( np.linalg.inv(hessian) ,   np.matmul(A_new.T, b_vect_new)   )
        p_delta = p_delta.flatten()
        # print('p_delta shape: ',p_delta.shape)

        p += p_delta

        p_delta_norm = np.sqrt(p_delta[0]**2 + p_delta[1]**2)
        print('image error: ', b.sum(), ', p_delta: ', p_delta, ', norm: ', p_delta_norm)

    return p