# convert Color to BW video clips

import os
import argparse

import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='*', help='Filename of input video')
    parser.add_argument('--input_dir', type=str, default='data/raw/', help='Directory of input files')
    parser.add_argument('--output_dir', type=str, default='data/converted/', help='Directory of output files')
    parser.add_argument('--out_dim', type=int, nargs=2, default=None, help='Dimensions of output frames (width, height)')
    parser.add_argument('--fps', type=int, default=None, help='Number of fps of output files')

    args = parser.parse_args()
    return args

def parse_config(args):
    with open('config.yml', 'r') as f:
        config = yaml.load(f)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    return dict2namespace(config)
    
def color2bw(inputname, inputpath, outputpath, out_dim, fps):
    if inputname.endswith(".mp4"):
        
        # store informations about the original video
        cap = cv2.VideoCapture(inputpath + inputname)
        # original dimensions
        width, height = int(cap.get(3)), int(cap.get(4))

        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v');
        
        # parameters of output file
        if out_dim == None:
            # dimensions of the output image
            new_width, new_height = width, height
        else:
            new_width, new_height = out_dim
        if fps == None:
            # number of frames
            fps = 30.0
    
        # grayscale output video
        gray_out = cv2.VideoWriter(
            outputpath + 'bw_' + inputname,
            fourcc,
            fps,
            (new_width, new_height),
            isColor=False
        )
        
        # color output video
        color_out = cv2.VideoWriter(
            outputpath + 'color_' + inputname,
            fourcc,
            fps,
            (new_width, new_height),
            isColor=True
        )


        while(cap.isOpened()):
            ret, frame = cap.read()
            # check if we are not at the end of the video
            if ret==True:
                
                #resize frame
                frame = cv2.resize(frame, (new_width, new_height), interpolation = cv2.INTER_LINEAR)
                
                # write the color frame
                color_out.write(frame)
                
                # change color to BW
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # write the grayscaled frame
                gray_out.write(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            # end of the video
            else:
                break

        # release everything if job is finished
        cap.release()
        gray_out.release()
        color_out.release()

def main():
    args = parse_args()

    if args.filename == '*':
        for filename in os.listdir(args.input_dir):
            color2bw(inputname = filename, inputpath = args.input_dir, outputpath = args.output_dir, out_dim = args.out_dim, fps = args.fps)
    else:
        color2bw(inputname = args.filename, inputpath = args.input_dir, outputpath = args.output_dir, out_dim = args.out_dim, fps = args.fps)
        
    # cleanup
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__':
    main()
