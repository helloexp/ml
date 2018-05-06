# coding:utf-8
import skvideo.io
import cv2
import numpy as np
import time

def remove_wartermark(input, output):
    mask = cv2.imread("../mask/mask.jpg", cv2.IMREAD_GRAYSCALE)
    src = cv2.imread(input)
    img = cv2.inpaint(src, mask, 3, cv2.INPAINT_NS)
    cv2.imwrite(output, img)


def build_mask(path):
    original_img = cv2.imread(path)

    height, width, channel = original_img.shape
    print(height, width, channel)
    center_x = int(width / 2)
    center_y = int(height / 2)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    bgr = [140, 150, 181]
    val = np.array(bgr)
    upper = val + 20
    low = upper - 20
    img = cv2.inRange(original_img, low, upper)

    cut_x = 40
    cut_y = 50
    start_x = center_x - cut_x
    end_x = center_x + cut_x
    start_y = center_y - cut_y
    end_y = center_y + cut_y

    img = img[start_y:end_y, start_x:end_x]

    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    res_img = np.zeros((height, width))
    res_img[start_y:end_y, start_x:end_x] = img

    cv2.imwrite("../mask/mask.jpg", res_img)
    return res_img


def remove_video_wartermark(input, output):
    cap = cv2.VideoCapture(input)

    # In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable.
    # MJPG results in high size video. X264 gives very small size video)
    # In Windows: DIVX (More to be tested and added)
    # In OSX :

    mask = cv2.imread("../mask/video_mask.jpg", cv2.IMREAD_GRAYSCALE)


    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_out = None

    while (cap.isOpened()):
        ret, frame = cap.read()

        if(frame is None):
            break

        frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_NS)

        if video_out is None:
            (h, w) = frame.shape[:2]
            video_out = cv2.VideoWriter(output, fourcc, 25, (w, h), True)

        if (ret):
            # flip = cv2.flip(frame, 0)

            video_out.write(frame)
        else:
            break

    cap.release()
    video_out.release()



# ffmpeg -i project/wartermark/res.avi -c:v libx264 project/wartermark/convert.mp4
# ffmpeg -i yourvideo.mp4 -vn -ab 128k outputaudio.mp3
# ffmpeg -i a.wav  -i a.avi out.avi
def write_video(input, output):
    mask = cv2.imread("../mask/video_mask.jpg", cv2.IMREAD_GRAYSCALE)
    inputparameters = {}
    outputparameters = {}

    reader = skvideo.io.FFmpegReader(input, inputdict=inputparameters,
                                     outputdict=outputparameters)

    # skvideo.io.vread()
    writer = skvideo.io.FFmpegWriter(output)

    inputframenum = reader.inputframenum
    print(inputframenum)

    for frame in reader.nextFrame():
        # frame = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)

        b, g, r = cv2.split(frame)
        frame = cv2.merge([r, g, b])

        # cv2.imshow("frame",frame)
        # cv2.waitKey(0)
        # kernel = np.ones((2, 2), np.uint8)
        # frame = cv2.dilate(frame, kernel, iterations=3)

        writer.writeFrame(frame)

    reader.close()
    writer.close()


def generate_video_mask(input, output):
    src = cv2.imread(input)
    mask_little = cv2.imread("../mask/mask_little.jpg", cv2.IMREAD_GRAYSCALE)

    print(mask_little.shape)
    height, width = mask_little.shape

    center_x = 1162
    center_y = 600

    video_mask = np.zeros((src.shape[0], src.shape[1]))
    video_mask[center_y - int(height / 2):center_y + int(height / 2),
    center_x - int(width / 2):center_x + int(width / 2)] = mask_little

    cv2.imwrite(output, video_mask)


def generate_video_mask2(input, output):
    src = cv2.imread(input)

    center_x = 1162
    center_y = 600

    cut_x = 70
    cut_y = 120
    start_x = center_x - cut_x
    end_x = center_x + cut_x
    start_y = center_y - cut_y
    end_y = center_y + cut_y

    bgr = [210, 210, 210]
    val = np.array(bgr)
    upper = val + 20
    low = upper - 25
    img = cv2.inRange(src, low, upper)
    video_mask_little = img[start_y:end_y, start_x:end_x]
    # cv2.imshow("frame",video_mask_little)
    # cv2.waitKey(0)

    kernel = np.ones((3, 3), np.uint8)
    video_mask_little = cv2.dilate(video_mask_little, kernel, iterations=1)

    video_mask = np.zeros((src.shape[0], src.shape[1]))
    video_mask[start_y:end_y, start_x:end_x] = video_mask_little

    cv2.imwrite(output,video_mask)

def mix_sound_video(video_sound,video_not_sound):
    time_time = time.time()

    rfind = video_sound.rfind(os.sep)+1


    out_rfind = video_not_sound.rfind(os.sep)+1
    base_output_dir = video_not_sound[0:out_rfind]

    mp3_dir = base_output_dir + str(int(time_time))+".mp3"

    video_not_sound_mp4=base_output_dir+str(int(time_time+1)) + ".mp4"
    finally_mp4=base_output_dir+video_sound[rfind:]

    make_sound_str="ffmpeg -i %s -vn -ab 128k %s" % (video_sound, mp3_dir)

    convert_avi_to_mp4="ffmpeg -i %s -c:v libx264 %s" % (video_not_sound,video_not_sound_mp4)

    mix_sound="ffmpeg -i %s  -i %s %s" % (video_not_sound_mp4,mp3_dir,finally_mp4)
    os.system(make_sound_str)
    os.system(convert_avi_to_mp4)
    os.system(mix_sound)

    os.remove(video_not_sound)
    os.remove(mp3_dir)
    os.remove(video_not_sound_mp4)

    # ffmpeg -i project/wartermark/res.avi -c:v libx264 project/wartermark/convert.mp4
    # ffmpeg -i yourvideo.mp4 -vn -ab 128k outputaudio.mp3
    # ffmpeg -i a.wav  -i a.avi out.avi


import sys,os

# tar -cvf wartermark.tar.gz *
# pip install sk-video
if __name__ == '__main__':
    # path = sys.argv[1]
    # path="/Users/tong/PycharmProjects/machinelearning/project/wartermark/1499838685844.jpg"
    # mask=build_mask(path)


    # print(mask.shape)
    # output="../wartermark/res_test.jpg"
    # remove_wartermark(path,output)

    # cv2.imshow("hst", img)
    # cv2.waitKey(0)

    # write_video(video_path,"./res.mp4")

    video_base_path = "/Users/tong/Downloads/work/full/"
    output = "/Users/tong/Downloads/work/full2/"

    count = 0

    error=[]

    for f in os.listdir(video_base_path):
        count+=1
        if(count%10==0):
            print((count,f))
        file_path = video_base_path + f
        output_path = output + f

        try:
            if(f.endswith("jpg")):
                remove_wartermark(file_path, output_path)
            else:
                avi_file = output_path.replace(".mp4", ".avi")
                remove_video_wartermark(file_path, avi_file)
                mix_sound_video(file_path,avi_file)
        except BaseException:
            error.append(f)
            # remove_video_wartermark(video_path, output)

            # mix_sound_video(video_path,output)


            # generate_video_mask2("../mask/video.jpg", "../mask/video_mask.jpg")
    print(error)