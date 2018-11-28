import os
import sys
import argparse

import cv2
import numpy as np
from skimage import img_as_float
import skimage.color as color
import scipy.ndimage.interpolation as sni
import caffe

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='*', help='Filename of input BW video')
    parser.add_argument('--input_dir', type=str, default='/home/ubuntu/Automatic-Video-Colorization/data/examples/converted/', help='Directory of input files')
    parser.add_argument('--output_dir', type=str, default='/home/ubuntu/Automatic-Video-Colorization/data/examples/recolorized/', help='Directory of output files')
    parser.add_argument('--gpu', dest='gpu', help='gpu id', type=int, default=0)
    parser.add_argument('--prototxt',dest='prototxt',help='prototxt filepath', type=str, default='./models/colorization_deploy_v2.prototxt')
    parser.add_argument('--caffemodel',dest='caffemodel',help='caffemodel filepath', type=str, default='./models/colorization_release_v2.caffemodel')

    args = parser.parse_args()
    return args
    
def image_colorization(frame, args):

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu)

	# Select desired model
	net = caffe.Net(args.prototxt, args.caffemodel, caffe.TEST)

	(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
	(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

	pts_in_hull = np.load('./resources/pts_in_hull.npy') # load cluster centers
	net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel
	# print 'Annealed-Mean Parameters populated'

	# load the original image
	img_rgb = img_as_float(frame).astype(np.float32)

	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,0] # pull out L channel
	(H_orig,W_orig) = img_rgb.shape[:2] # original image size

	# create grayscale version of image (just for displaying)
	img_lab_bw = img_lab.copy()
	img_lab_bw[:,:,1:] = 0
	img_rgb_bw = color.lab2rgb(img_lab_bw)

	# resize image to network input size
	img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
	img_lab_rs = color.rgb2lab(img_rs)
	img_l_rs = img_lab_rs[:,:,0]

	net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
	net.forward() # run network

	ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
	ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
	img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
	img_rgb_out = (255*np.clip(color.lab2rgb(img_lab_out),0,1)).astype('uint8') # convert back to rgb

	return img_rgb_out

def bw2color(args, inputname, inputpath, outputpath):
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
        
        while(cap.isOpened()):
            ret, frame_in = cap.read()
            # check if we are not at the end of the video
            if ret==True:                
                # convert BGR to RGB convention
                frame_in = frame_in[:,:,::-1]
                # colorize the BW frame
                frame_out = image_colorization(frame_in, args)
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
    args = parse_args()

    if args.filename == '*':
        for filename in os.listdir(args.input_dir):
            bw2color(args, inputname = filename, inputpath = args.input_dir, outputpath = args.output_dir)
    else:
        bw2color(args, inputname = args.filename, inputpath = args.input_dir, outputpath = args.output_dir)
        
    # cleanup
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
