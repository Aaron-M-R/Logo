# Loading the necessary packages 
import numpy as np
import cv2
import os
from imutils.object_detection import non_max_suppression
import pytesseract
from tqdm import tqdm
from matplotlib import pyplot as plt

'''source: https://nanonets.com/blog/deep-learning-ocr/'''
'''frozen east text detection: https://github.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/blob/master/frozen_east_text_detection.pb'''



'''Make a copy of image called `orig`'''
def make_orig(args):

    return orig



'''Creates blob and returns it along with original image and ratios of height and weight'''
def make_blob(args):
    
    image = cv2.imread(args['image'])

    #Saving a original image and shape
    orig = image.copy()

    (origH, origW) = image.shape[:2]

    # set the new height and width to default 320 by using args #dictionary.  
    (newW, newH) = (args["width"], args["height"])

    #Calculate the ratio between original and new image for both height and weight. 
    #This ratio will be used to translate bounding box location on the original image. 
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the original image to new dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    return blob, orig, rW, rH



'''Returns a bounding box and probability score if it is more than minimum confidence'''
def predict_boxes(prob_score, geo, args):
    (numR, numC) = prob_score.shape[2:4]
    boxes = []
    confidence_val = []

    # loop over rows
    for y in range(0, numR):
        scoresData = prob_score[0, 0, y]
        x0 = geo[0, 0, y]
        x1 = geo[0, 1, y]
        x2 = geo[0, 2, y]
        x3 = geo[0, 3, y]
        anglesData = geo[0, 4, y]

        # loop over the number of columns
        for i in range(0, numC):
            if scoresData[i] < args["min_confidence"]:
                continue

            (offX, offY) = (i * 4.0, y * 4.0)

            # extracting the rotation angle for the prediction and computing the sine and cosine
            angle = anglesData[i]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # using the geo volume to get the dimensions of the bounding box
            h = x0[i] + x2[i]
            w = x1[i] + x3[i]

            # compute start and end for the text pred bbox
            endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
            endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
            startX = int(endX - w)
            startY = int(endY - h)

            boxes.append((startX, startY, endX, endY))
            confidence_val.append(scoresData[i])

    # return bounding boxes and associated confidence_val
    return boxes, confidence_val



'''Find predictions and  apply non-maxima suppression''' 
def make_boxes(scores, geometry, args):
    (boxes, confidence_val) = predict_boxes(scores, geometry, args)
    boxes = non_max_suppression(np.array(boxes), probs=confidence_val)
    return boxes



'''Text Detection and Recognition'''
def detect_text(scores, geometry, args):

    # initialize the list of results
    results = []

    boxes = make_boxes(scores, geometry, args)
    blob, orig, rW, rH = make_blob(args)

    # loop over the bounding boxes to find the coordinate of bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        #extract the region of interest
        r = orig[startY:endY, startX:endX]

        #configuration setting to convert image to string.  
        configuration = ("-l eng --oem 1 --psm 8")
        ##This will recognize the text from the image of bounding box
        text = pytesseract.image_to_string(r, config=configuration)

        # append bbox coordinate and associated text to the list of results 
        results.append(((startX, startY, endX, endY), text))
        return results


def extract_text_cnn(orig_path):
    
    '''TODO: write description of pass_blob'''
    def pass_blob(blob, net):
        net.setInput(blob)
        scores, geometry = net.forward(layerNames)
        return scores, geometry

    east_path = 'frozen_east_text_detection.pb'

    # Creating argument dictionary for the default arguments needed in the code. 
    args = {
        "image":orig_path, 
        "east": east_path, 
        "min_confidence":0.5, 
        "width":320, 
        "height":320}
    
    # load the pre-trained EAST model for text detection 
    net = cv2.dnn.readNet(args["east"])

    # We would like to get two outputs from the EAST model. 
    # 1. Probabilty scores for the region whether that contains text or not. 
    # 2. Geometry of the text -- Coordinates of the bounding box detecting a text
    # The following two layer need to pulled from EAST model for achieving this. 
    layerNames = ["feature_fusion/Conv_7/Sigmoid",
                  "feature_fusion/concat_3"]

    # Forward pass the blob from the image to get the desired output layers
    blob, orig, rW, rH = make_blob(args)

    # Get scores and geometry by passing blob
    scores, geometry = pass_blob(blob, net)

    #Display the image with bounding box and recognized text
    orig_image = orig.copy()

    # Emtpy array for storing text
    all_text = np.array([])

    # Detect text in image
    results = detect_text(scores, geometry, args)

    # Moving over the results and display on the image
    if results is not None:
        for ((start_X, start_Y, end_X, end_Y), text) in results:

            # Displaying text
            text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
            cv2.rectangle(orig_image, (start_X, start_Y), (end_X, end_Y),
                (0, 0, 255), 2)
            cv2.putText(orig_image, text, (start_X, start_Y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

            # Recording text
            all_text = np.append(all_text, text)
        
    return all_text

# plt.imshow(orig_image)
# plt.title('Output')
# plt.show()



