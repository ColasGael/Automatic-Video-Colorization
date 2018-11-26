import cv2
import numpy as np
import glob


def main():
    videos = glob.glob("./baking/*")
    
    count = 0
    for i in range(len(videos)):
        
        #'ScotlandTransport.mp4'
        
        vidcap = cv2.VideoCapture(videos[i])
        success,image = vidcap.read()
        resized_image_new = cv2.resize(image, (256, 256)) 
        count = 0
        while success:
        
          resized_image_old = resized_image_new     
          success,image = vidcap.read()
          if success:
              resized_image_new = cv2.resize(image, (256, 256)) 
              img = np.concatenate((resized_image_new, resized_image_old), 2)
          print('Read a new frame: ', success)
          #cv2.imwrite("frame%d.jpg" % count, img)     # save frame as JPEG file 
          if count % 2 ==0:

              np.save("./frames2/video{}_frame{}".format(i, count), img)
          count += 1
          
if __name__ == "__main__":
    main()
          
          
          
          
