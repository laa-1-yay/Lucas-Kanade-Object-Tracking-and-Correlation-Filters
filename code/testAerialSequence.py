import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from SubtractDominantMotion import SubtractDominantMotion

# write your script here, we recommend the above libraries for making your animation

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
plt.ion()

def disp_patch_image(frame, i):
    ax1.clear()
    ax1.imshow(frame)
    title =  'Frame: ' + str(i)
    ax1.set_title(title)
    plt.pause(0.000001)

if __name__ == '__main__':

    video_frames = np.load('../data/aerialseq.npy')
    print(video_frames.shape)

    output_frames = []

    for i in range(0, video_frames.shape[2]-1):
        print('-------------------------------------  ', i+1, '  -------------------------------------------')
        image1 = video_frames[:,:,i]
        image2 = video_frames[:,:,i+1]

        mask = SubtractDominantMotion(image1, image2)

        alpha = np.ones(image2.shape)
        frame = np.dstack( (image2, image2, image2, alpha))

        frame[mask] += [0.3,0.7,0, 0.02]
        output_frames.append(frame)
        disp_patch_image(frame, i)

    output_frames = np.asarray(output_frames)

    frame_list = [30, 60, 90, 120]
    f, axarr = plt.subplots(1, len(frame_list))

    for i in range(len(frame_list)):
        axarr[i].imshow(output_frames[frame_list[i]] ,  cmap="gray")
        axarr[i].axis('off')
    plt.subplots_adjust(wspace=0.025, hspace=0)

    plt.savefig('../lbahl/3_3.png', bbox_inches='tight',  dpi = 1000)