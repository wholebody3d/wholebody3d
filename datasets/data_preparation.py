import cv2
import os

def convert_mp4_to_image(inpath, outpath, each_x_frame=1):
    print("load "+inpath)
    vidcap = cv2.VideoCapture(inpath)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % each_x_frame == 0:
            cv2.imwrite(outpath+str(count).zfill(4)+".jpg", image)  # save frame as JPEG file
        success, image = vidcap.read()
        if success:
            count += 1
            if count % 100 == 0:
                print('Finish frame: ', count)
                # time.sleep(1)
    print("Finish all ", count, " images")


def convert_h36m_mp4_to_image(base_path, each_x_frame=1):
    subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    # subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    for subject in subjects:
        inpath_base = base_path+subject+"/Videos"
        outpath_base = base_path+subject+"/Images"
        if not os.path.exists(outpath_base):
            os.makedirs(outpath_base)
        videos = os.listdir(inpath_base)
        for video in videos:
            inpath = inpath_base + "/" + video
            outpath = outpath_base + "/" + video[:-4]
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            outpath = outpath + "/frame_"
            convert_mp4_to_image(inpath, outpath, each_x_frame)

if __name__ == "__main__":
    path = "./"
    convert_h36m_mp4_to_image(path+'Human36m/')
