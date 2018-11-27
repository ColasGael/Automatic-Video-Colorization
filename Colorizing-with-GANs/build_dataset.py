"""Split the SIGNS dataset into train/dev/test and resize images to 64x64.

The SIGNS dataset comes in the following format:
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...

Original images have size (3024, 3024).
Resizing to (64, 64) reduces the dataset size from 1.16 GB to 4.7 MB, and loading smaller images
makes training faster.

We already have a test set created, so we only need to split "train_signs" into train and dev sets.
Because we don't have a lot of images and we want that the statistics on the dev set be as
representative as possible, we'll take 20% of "train_signs" as dev set.
"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm
import cv2

# size of the resized frames
SIZE = 256

# subfolder of the "Moments_in_Time" dataset to consider
SUBFOLDER = "/baking"

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='home/ubuntu/Automatic-Video-Colorization/data/Moments_in_Time_Mini', help="Directory with the Moments in Time dataset")
parser.add_argument('--output_dir', default='home/ubuntu/Automatic-Video-Colorization/data/momentsintime', help="Where to write the new data")


def split_resize_and_save(filename, output_dir, size=SIZE):
    """Split the video clip in pair of consecutive frames, resize the frames, and save the pairs to the `output_dir`"""
 
    # counter to build pairs of consecutive frames
    count = 0
                
    vidcap = cv2.VideoCapture(filename)
    
    success, image = vidcap.read()
    # default : use bilinear interpolation
    frame = cv2.resize(image, (size, size)) 
    
    while success:
    
      frame_prev = frame     
      success, frame = vidcap.read()
      # convert BGR to RGB convention
      frame = frame[:,:,::-1]
      
      if success:
          frame = cv2.resize(frame, (size, size)) 
          img = np.concatenate((frame, frame_prev), 2)
          
      print('Read a new frame: ', success)
      
      #cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file 
      
      if count % 2 ==0:
          np.save(output_dir + "/video{}_frame{}".format(i, count), img)
          
      count += 1

if __name__ == '__main__':
    args = parser.parse_args()
    
    if not SUBFOLDER == None:
        args.data_dir = args.data_dir + SUBFOLDER
    
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'training')
    test_data_dir = os.path.join(args.data_dir, 'validation')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(train_data_dir)
    filenames = [os.path.join(train_data_dir, f) for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [os.path.join(test_data_dir, f) for f in test_filenames if f.endswith('.jpg')]

    # Split the images in 'train_moments' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed so that the split is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.9 * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_moments'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))

        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            split_resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
