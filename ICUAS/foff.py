import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import time

#convert RBG to YIQ
def rgb2ntsc(src):
    [rows,cols]=src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#convert YIQ to RBG
def ntsc2rbg(src):
    [rows, cols] = src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#Build Gaussian Pyramid
def build_gaussian_pyramid(image, level=3):
    pyramid = [image]
    for i in range(level - 1):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

def build_laplacian_pyramid(image, level=3):
    gaussian_pyramid = build_gaussian_pyramid(image, level)
    pyramid = []
    for i in range(level - 1):
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1])
        if expanded.shape[0] > gaussian_pyramid[i].shape[0] or expanded.shape[1] > gaussian_pyramid[i].shape[1]:
            expanded = cv2.resize(expanded, (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0]))
        diff = cv2.subtract(gaussian_pyramid[i], expanded)
        pyramid.append(diff)
    pyramid.append(gaussian_pyramid[-1])
    return pyramid

#load video from file
def load_video(video_filename):
    cap=cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    video_tensor=np.zeros((frame_count,height,width,3),dtype='uint8')
    x=0
    while cap.isOpened():
        print(x)
        ret,frame=cap.read()
        if ret is True:
            video_tensor[x]=frame
            x+=1
        else:
            break
    return video_tensor,fps

# apply temporal ideal bandpass filter to gaussian video
def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor,levels=3):
    print("Gaussiating video...")
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr
        del pyr
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
        vid_data[i]=gaussian_frame

    print("Gaussiating video done :)")
    return vid_data

#amplify the video
def amplify_video(gaussian_vid,amplification=50):
    return gaussian_vid*amplification

#reconstract video from original video and gaussian video
def reconstract_video(amp_video,origin_video,levels=3):
    print("Reconstructing video...")
    final_video=np.zeros(origin_video.shape, dtype='uint8')
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img=img+origin_video[i]
        final_video[i]=img
    
    print("Reconstructing video done :)")

    return final_video

#save video to files
def save_video(video_tensor, filename='out.avi'):
    print("Saving video...")
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter(filename, fourcc, 30, (width, height), 1)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()

#magnify color
def magnify_color(video_name,low,high,levels=3,amplification=20, filename='out.avi'):
    t,f=load_video(video_name)
    print("Video loaded")
    gau_video=gaussian_video(t,levels=levels)
    filtered_tensor=temporal_ideal_filter(gau_video,low,high,f)
    print("Filtering Done")
    amplified_video=amplify_video(filtered_tensor,amplification=amplification)
    final=reconstract_video(amplified_video,t,levels=levels)
    save_video(final, filename=filename)

#build laplacian pyramid for video
def laplacian_video(video_tensor,levels=3):
    print("laplaciating video...")
    tensor_list=[]
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_laplacian_pyramid(frame,level=levels)
        if i==0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
    print("laplaciating video done :)")
    return tensor_list

#butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y
# ...
def reconstract_from_tensorlist(filter_tensor_list,levels=3):
    # Initialize final with the shape of the last level of filter_tensor_list
    final_shape = filter_tensor_list[-1].shape
    final = np.zeros((final_shape[0], final_shape[1]*2**(levels-1), final_shape[2]*2**(levels-1), final_shape[3]))

    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up = cv2.pyrUp(up)
            next_level_up = cv2.pyrUp(filter_tensor_list[n + 1][i])
            if up.shape != next_level_up.shape:
                height, width, _ = up.shape
                next_level_up = cv2.resize(next_level_up, (width, height))
            upOG = up + next_level_up

        # Resize upOG to the shape of final[i] before assigning it to final[i]
        height, width, _ = final[i].shape
        upOG = cv2.resize(upOG, (width, height))

        final[i] = upOG
    return final
# ...

#manify motion
def magnify_motion(video_name,low,high,levels=3,amplification=20):
    t,f=load_video(video_name)
    lap_video_list=laplacian_video(t,levels=levels)
    filter_tensor_list=[]
    for i in range(levels):
        filter_tensor=butter_bandpass_filter(lap_video_list[i],low,high,f)
        filter_tensor*=amplification
        filter_tensor_list.append(filter_tensor)
    recon=reconstract_from_tensorlist(filter_tensor_list)
    final=t+recon
    save_video(final)


magnify_motion("baby.mp4",0.4,3, amplification=20)