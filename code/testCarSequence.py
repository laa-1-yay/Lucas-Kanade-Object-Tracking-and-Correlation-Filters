import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
from scipy.interpolate import RectBivariateSpline

# write your script here, we recommend the above libraries for making your animation

rect = np.asarray([59, 116, 145, 151])

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.ion()

def disp_patch_image(frame, rect, i):
    ax1.clear()
    ax1.imshow(frame, cmap='gray')
    rect = patches.Rectangle(rect[:2], rect[2] - rect[0]+1, rect[3] - rect[1]+1, linewidth=1, edgecolor='y', fill=False)
    ax1.add_patch(rect)
    title =  'Frame: ' + str(i)
    ax1.set_title(title)
    plt.pause(0.000001)

def get_template_from_img(img, rect):
    ###################  CALCULATING SPLINE AND INTERPOLATING IMAGE WARP #########################
    spline_It = RectBivariateSpline( np.arange(img.shape[0]), np.arange(img.shape[1]), img)

    template = spline_It.__call__( np.linspace(rect[1], rect[3], np.round(rect[3] - rect[1]+1), endpoint=True),
                                   np.linspace(rect[0], rect[2], np.round(rect[2] - rect[0]+1), endpoint=True), grid=True)
    return template


if __name__ == '__main__':

    video_frames = np.load('../data/carseq.npy')
    print(video_frames.shape)

    p_res = np.zeros(2)

    rect_arr = []
    rect_arr.append(rect)

    rect_new = rect.copy()
    template_img = get_template_from_img(video_frames[:, :, 0], rect)

    for i in range(1, video_frames.shape[2]):
        print('-------------------------------------  ', i, '  -------------------------------------------')
        p = LucasKanade( template_img, video_frames[:,:,i], rect_new, np.zeros(2) )
        print('p: ', p)
        rect_new =  rect_new + np.asarray([ p[0], p[1], p[0], p[1]])
        rect_arr.append(rect_new)
        disp_patch_image(video_frames[:,:,i], rect_new, i)
        template_img =  get_template_from_img( video_frames[:,:,i], rect_new)

    rect_arr = np.asarray(rect_arr)
    np.save('../data/carseqrects.npy', rect_arr)
    # rect_arr = np.load('../data/carseqrects.npy')

    frame_list = [1, 100, 200, 300, 400]
    f, axarr = plt.subplots(1, len(frame_list))

    for i in range(len(frame_list)):
        axarr[i].imshow(video_frames[:,:,frame_list[i]] ,  cmap="gray")
        rect = patches.Rectangle(rect_arr[frame_list[i], :2], rect_arr[frame_list[i], 2] - rect_arr[frame_list[i], 0] , rect_arr[frame_list[i], 3] - rect_arr[frame_list[i], 1], linewidth=0.6, edgecolor='y', fill=False)
        axarr[i].add_patch(rect)
        axarr[i].axis('off')
    plt.subplots_adjust(wspace=0.025, hspace=0)

    plt.savefig('../lbahl/1_3.png', bbox_inches='tight',  dpi = 1000)