import os
import sys
import argparse

import cv2
import numpy as np
from skimage import img_as_float
import skimage.color as color
import scipy.ndimage.interpolation as sni

import tensorflow as tf
from options import ModelOptions
from models import MomentsInTimeModel
    
def image_colorization_propagation(img_bw_in, img_rgb_prev, options):
    # reset tensorflow graph
    tf.reset_default_graph()
    print(img_bw_in.shape, img_rgb_prev.shape)
    # create a session environment
    with tf.Session() as sess:
        model = MomentsInTimeModel(sess, options)

        # build the model and initialize
        model.build()
        sess.run(tf.global_variables_initializer())

        # load model only after global variables initialization
        model.load()

        # colorize the image based on the previous one
        feed_dic = {model.input_rgb: np.expand_dims(img_bw_in, axis=0), model.input_rgb_prev: np.expand_dims(img_rgb_prev, axis=0)}
        fake_image = sess.run(model.sampler, feed_dict=feed_dic)
        img_rgb_out = postprocess(tf.convert_to_tensor(fake_image), colorspace_in=options.color_space, colorspace_out=COLORSPACE_RGB)
        img_rgb_out = img_rgb_out.squeeze(0)

    return img_rgb_out

def bw2color(options, inputname, inputpath, outputpath):
    if inputname.endswith(".mp4"):
        
        # store informations about the original video
        cap = cv2.VideoCapture(inputpath + inputname)
        # original dimensions
        width, height = int(cap.get(3)), int(cap.get(4))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v');
        # parameters of output file
            # dimensions of the output image
        new_width, new_height = width, height
            # number of frames
        fps = 30.0
    
        # recolorized output video
        color_out = cv2.VideoWriter(
            outputpath + 'color_' + inputname,
            fourcc,
            fps,
            (new_width, new_height),
            isColor=True
        )
        
        # TO CHANGE pick the first frame from the original video clip 
        cap_temp = cv2.VideoCapture("/home/ubuntu/Automatic-Video-Colorization/data/examples/raw/" + inputname[3:])
        ret, frame_prev = cap_temp.read()
        
        while(cap.isOpened()):
            ret, frame_in = cap.read()
            # check if we are not at the end of the video
            if ret==True:                
                # convert BGR to RGB convention
                frame_in = frame_in[:,:,::-1]
                # colorize the BW frame
                frame_out = image_colorization_propagation(frame_in, frame_prev, options)
                # save the recolororized frame
                frame_prev = frame_out
                # convert RGB to BGR convention
                frame_out = frame_out[:,:,::-1]
                # write the color frame
                color_out.write(frame_out)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # end of the video
            else:
                break

        # release everything if job is finished
        cap.release()
        color_out.release()

def main():
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
