# Python code for Background subtraction using OpenCV
# https://www.geeksforgeeks.org/python-background-subtraction-using-opencv/
# https://tv.ring.com/detail/videos/
# https://www.geeksforgeeks.org/python-program-extract-frames-using-opencv/
# https://iq.opengenus.org/connected-component-labeling/
# https://www.pyimagesearch.com/2021/02/22/opencv-connected-component-labeling-and-analysis/
# https://www.geeksforgeeks.org/python-opencv-cv2-rectangle-method/


import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import sys





(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

def track():
    # Set up tracker.
    # Instead of CSRT, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        elif tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        elif tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        elif tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        elif tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        elif tracker_type == 'GOTURN':
             tracker = cv2.TrackerGOTURN_create()
        elif tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        elif tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()

        # Read video
        video = cv2.VideoCapture("ring1.mov")
        #video = cv2.VideoCapture(0) # for using CAM

        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()

        # Read first frame.
        ok, frame = video.read()
        if not ok:
            print ('Cannot read video file')
            sys.exit()

        # # Define an initial bounding box
        # bbox = (287, 23, 86, 3200)

        # Uncomment the line below to select a different bounding box
        bbox = cv2.selectROI(frame, False)

        # Initialize tracker with first frame and bounding box
        ok = tracker.init(frame, bbox)

        # fgbg = cv2.createBackgroundSubtractorMOG2()

        while True:
            # Read a new frame
            ok, frame = video.read()
            if not ok:
                break

            # apply background subtraction to each frame
      
            # Start timer
            timer = cv2.getTickCount()

            # Update tracker
            ok, bbox = tracker.update(frame)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Draw bounding box
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
        
            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
            # Display result
            cv2.imshow("Tracking", frame)

            # Exit if ESC pressed
            if cv2.waitKey(1) & 0xFF == ord('q'): # if press SPACE bar
                break

        video.release()
        cv2.destroyAllWindows()




parent = [0] * 400000000

def unique(image):
    uniqueVals = [0] * 10000

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if image[r][c] not in uniqueVals:
                uniqueVals.append(image[r][c])

    return uniqueVals





def labelColorize(labelImage):
    rows = labelImage.shape[0]
    cols = labelImage.shape[1]

    outputImage = np.empty_like(labelImage)

    uniqueLabels = unique(labelImage)
    colorMap = [0] * len(uniqueLabels)

    for label in uniqueLabels:
        if label != 0:
            color = np.random.uniform(0, 255)
            colorMap[label] = color
        else:
            colorMap[label] = 255

    for i in rows:
        for j in cols:
            color = colorMap[labelImage[i][j]]
            outputImage[i][j] = color

    return outputImage









def shapeProperties(labelImage):
    taken = [0] * 100

    inArray = False
    totalLabels = 0
    i = 0
    area = 0

    for r in labelImage.shape[0]:
        for c in labelImage.shape[1]:
            if labelImage[r][c] != 255:
                for i in range(100):
                    if labelImage[r][c] in taken:
                        inArray = True
                        break

            if inArray == False:
                taken[totalLabels] = labelImage[r][c]
                totalLabels += 1
            else:
                inArray == False

    print("**************************************************************************")
    i = 0
    for i in range(100):
        if taken[i] != 0:
            for r in labelImage.shape[0]:
                for c in labelImage.shape[1]:
                    if labelImage[r][c] == taken[i]:
                        area += 1

            print("\n\nLabel: %d ", taken[i])
            print("\nArea: %d", area)

    print("\n\n\n**************************************************************************")
    print("\nNumber of Connected Components: %d", totalLabels)









def findFunction(X):
    j = X
    while(parent[j] != 0):
        j = parent[j]
    return j





def unionFunction(X, Y):
    j = findFunction(X)
    k = findFunction(Y)
    if(j != k):
        parent[k] = j





def labelRegion(img_bw, H, W):
    L = [[0]*W]*H
    newLabel = 1
    labelImage = img_bw

    for r in range(H):
        for c in range(W):
            if img_bw[r][c] == 255:
                if r >  0 and c > 0 and r < H and c < W:
                    pa = L[r-1][c-1]
                else:
                    pa = 0

                if r >  0 and c > 0 and r < H and c < W:
                    pb = L[r-1][c]
                else:
                    pb = 0

                if r >  0 and c > 0 and r < H and c < W-1:
                    pc = L[r-1][c+1]
                else:
                    pc = 0

                if r >  0 and c > 0 and r < H and c < W:
                    pd = L[r][c-1]
                else:
                    pd = 0



                if r-1<H and c<W and img_bw[r-1][c] == 255:
                    L[r][c] = pb
                else:
                    if r-1<H and c+1<W and img_bw[r-1][c+1] == 255:
                        if img_bw[r-1][c-1] != 0:
                            L[r][c] = pa
                            unionFunction(pa, pc)
                        else:
                            if img_bw[r][c-1] and img_bw[r][c-1] == 255:
                                L[r][c] = pc
                                unionFunction(pc, pd)
                            else:
                                L[r][c] = pc
                    else:
                        if img_bw[r-1][c-1] and img_bw[r-1][c-1] == 255:
                            L[r][c] = pa
                        else:
                            if img_bw[r][c-1] and img_bw[r][c-1] == 255:
                                L[r][c] = pd
                            else:
                                L[r][c] = newLabel
                                newLabel += 1
            else:
                L[r][c] = 0



    for r in range(H):
        for c in range(W):
            value = L[r][c]
            if findFunction(value) != 0:
                L[r][c] = findFunction(value)
            labelImage[r][c] = L[r][c]
            print(labelImage[r][c])

    return labelImage






def connected_component_label(path):
    # Getting the input image (grayscale because background subtracted)
    img = cv2.imread(path, 0)
    # Converting those pixels with values 1-50 to black and others to 1
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]

    # apply connected component analysis to the thresholded image
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    # labelImage = labelRegion(img, img.shape[0], img.shape[1])
    # labelImage = labelColorize(labelImage)

    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    # Showing Original Image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Orginal Image")
    plt.show()
    
    #Showing Image after Component Labeling
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    plt.show()

    maxArea = 0
    savedLabel = 0

    # loop over the number of unique connected component labels
    for i in range(1, numLabels):
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        # find the greatest one- eliminating noise
        if area > maxArea:
            savedLabel = i
            savedX = x
            savedY = y
            savedW = w
            savedH = h
            savedCX = cX
            savedCY = cY
            maxArea = area

    # clone our original image (so we can draw on it) and then draw
    # a bounding box surrounding the connected component along with
    # a circle corresponding to the centroid
    output = img.copy()
    cv2.rectangle(output, (savedX, savedY), (savedX + savedW, savedY + savedH), (255, 0, 0), 3)
    cv2.circle(output, (int(savedCX), int(savedCY)), 4, (0, 0, 255), -1)

    # construct a mask for the current connected component by
    # finding a pixels in the labels array that have the current
    # connected component ID
    componentMask = (labels == savedLabel).astype("uint8") * 255
    componentMask = cv2.rectangle(componentMask, (savedX, savedY), (savedX + savedW, savedY + savedH), (255, 0, 0), 3)

    # show our output image and connected component mask
    cv2.imshow("Output", output)
    cv2.imshow("Isolated Image", componentMask)
    cv2.waitKey(0)
    





def bg_sub(path):
    # Path to video file
    vidObj = cv2.VideoCapture(path)
    # Used as counter variable
    count = 0
    # checks whether frames were extracted
    success = 1
    # background subtraction
    fgbg = cv2.createBackgroundSubtractorMOG2()

    # for every frame in the video
    while success:
        
        # capture frame by frame
        success, frame = vidObj.read()

        # on every 50th frame
        if count % 50 == 0:
            # apply background subtraction to each frame
            fgmask = fgbg.apply(frame)
            
            # save the background subtracted frame with the frame count
            cv2.imwrite("frame%d.jpg" % count, fgmask)

            # call connected component labelling on the saved frame
            connected_component_label('frame%d.jpg' % count)

        # increment frame number
        count += 1

        # finish at the end of the video
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        
    # release the capture when everything is done
    vidObj.release()
    cv2.destroyAllWindows()



def main():
    bg_sub('ring3.mov')

if __name__ == "__main__":
    main()

