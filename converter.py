import numpy as np
import os
import cv2


for filename in os.listdir("data/raw"):
    if filename.endswith(".mp4"):

        cap = cv2.VideoCapture("data/raw/" + filename)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v');
        fps = 30.0
        out = cv2.VideoWriter(
            "data/converted/bw_" + filename,
            fourcc,
            fps,
            (int(cap.get(3)),int(cap.get(4))),
            isColor=False
        )


        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret==True:
                # Our operations on the frame come here
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # write the grayscaled frame
                out.write(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        # Release everything if job is finished
        cap.release()
        out.release()
cv2.destroyAllWindows()
