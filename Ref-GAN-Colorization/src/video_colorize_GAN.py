import os
import sys
import argparse

import cv2
import numpy as np
from PIL import Image
from skimage import img_as_ubyte, img_as_float
import skimage.color as color
import scipy.ndimage.interpolation as sni
from ops import postprocess
from ops import COLORSPACE_RGB, COLORSPACE_LAB

import tensorflow as tf
from options import ModelOptions
from models import MomentsInTimeModel

    
def image_colorization_propagation(model, img_bw_in, img_rgb_prev, img_rgb_first, options):

    # colorize the image based on the previous one
    feed_dic = {model.input_rgb: np.expand_dims(img_bw_in, axis=0), model.input_rgb_prev: np.expand_dims(img_rgb_prev, axis=0), model.input_rgb_first: np.expand_dims(img_rgb_first, axis=0)}
    fake_image, _ = model.sess.run([model.sampler, model.input_gray], feed_dict=feed_dic)
    fake_image = postprocess(tf.convert_to_tensor(fake_image), colorspace_in=options.color_space, colorspace_out=COLORSPACE_RGB)
    
    # evalute the tensor
    img_rgb_out = fake_image.eval()
    img_rgb_out = (img_rgb_out.squeeze(0) * 255).astype(np.uint8)

    return img_rgb_out

def bw2color(options, inputname, inputpath, outputpath):
    if inputname.endswith(".mp4"):
        # size of the input frames
        size = 256

        # check that the video exists
        path_to_video = os.path.join(inputpath, inputname)
        if not os.path.exists(path_to_video):
            print("The file :", path_to_video, "does not exist !")
        
        # store informations about the original video
        cap = cv2.VideoCapture(os.path.join(path_to_video))
        # original dimensions
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v');
        # parameters of output file
            # dimensions of the output image
        new_width, new_height = size, size
            # number of frames
        fps = 30.0
    
        # recolorized output video
        color_out = cv2.VideoWriter(
            os.path.join(outputpath, 'color_' + inputname),
            fourcc,
            fps,
            (new_width, new_height),
            isColor=True
        )
        
        # TO CHANGE to DL colorization of 1st frame
        # pick the first frame from the original video clip as the first reference
        cap_temp = cv2.VideoCapture(os.path.join(inputpath, "color" + inputname[2:]))
        
        ret_temp, frame_prev = cap_temp.read()
        # convert BGR to RGB convention
        frame_prev = frame_prev[:,:,::-1]
        frame_prev = cv2.resize(frame_prev, (size, size)) 
        # save the first frame as the reference
        frame_ref = frame_prev
        
        # count the number of recolorized frames
        frames_processed = 0

        with tf.Session() as sess:

            model = MomentsInTimeModel(sess, options)

            # build the model and initialize
            model.build()
            sess.run(tf.global_variables_initializer())

            # load model only after global variables initialization
            model.load()

            while(cap.isOpened()):
                ret, frame_in = cap.read()
                                
                # check if we are not at the end of the video
                if ret==True:                          
                    # convert BGR to RGB convention
                    frame_in = frame_in[:,:,::-1]
                    # resize the frame to match the input size of the GAN
                    frame_in = cv2.resize(frame_in, (size, size))

                    # colorize the BW frame
                    frame_out = image_colorization_propagation(model, frame_in, frame_prev, frame_ref, options)
                    
                    #generate sample
                    get_image = False
                    if get_image:                    
                        img = Image.fromarray(frame_out)

                        if not os.path.exists(model.samples_dir):
                            os.makedirs(model.samples_dir)

                        sample = model.options.dataset + "_" + inputname + "_" + str(frames_processed).zfill(5) + ".png"
                        img.save(os.path.join(model.samples_dir, sample))

                    # save the recolorized frame
                    frame_prev = frame_out
                    # convert RGB to BGR convention
                    frame_out = frame_out[:,:,::-1]
                    # write the color frame
                    color_out.write(frame_out)
                    
                    # print progress
                    frames_processed += 1
                    print("Processed {}/{} frames ({}%)".format(frames_processed, totalFrames, frames_processed * 100 //totalFrames), end="\r")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # end of the video
                else:
                    break

        # release everything if job is finished
        cap.release()
        color_out.release()

def main():

    # reset tensorflow graph
    tf.reset_default_graph()

    options = ModelOptions().parse()

    if options.filename == '*':
        for filename in os.listdir(options.input_dir):
            bw2color(options, inputname = options.filename, inputpath = options.input_dir, outputpath = options.output_dir)
    else:
        bw2color(options, inputname = options.filename, inputpath = options.input_dir, outputpath = options.output_dir)
        
    # cleanup
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
