import argparse
import os
import sys

import numpy as np
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate

def getFrame(vidcap, output, sec = 0, count = 0, frameRate = 0):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        cv2.imwrite(output + "/image"+str(count)+".jpg", image)     # save frame as JPG file
        print("\Extracted %i frames" % count, end="\r")
        return getFrame(vidcap, output, sec + frameRate, count + 1, frameRate)
    return count

def progress_bar(current, total, bar_length=20):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

    
def processFrames(count):
    processed = 5
    resultsFound = 0
    while processed < count:
        # run model
        # load image
        image_source, image = load_image("./output/image"+str(processed)+".jpg")
        boxes, logits, phrases = predict(
            model, image, keywords, box_threshold, text_threshold
        )
        
        if len(boxes) > 0:   
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite("./output/results/annotated_image"+str(processed)+".jpg", annotated_frame)
            resultsFound += len(boxes)
            
        processed += 1
        progress_bar(processed, count)
    
    return resultsFound

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO Search Engine", add_help=True)
    parser.add_argument("--input_video", "-i", type=str, required=True, help="path to video file")
    parser.add_argument("--keywords", "-k", type=str, required=True, help="keywords to find comma seperated")
    parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--framerate", type=float, default=0.5, help="frame extraction rate in seconds")
    parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")

    args = parser.parse_args()

    video = args.input_video
    keywords = args.keywords
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold
    framerate = args.framerate
    output = args.output
    
    if(False == os.path.isfile(video)):
        print("Video file doesn't exist!")
        exit(1);
    
    # Extract video frames
    vidcap = cv2.VideoCapture(video)
    
    print("Extracting video frames")
    
    frameCount = getFrame(vidcap, output, 0, 0, framerate)
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
    print("Loaded model")

    print("Processing frames")
    results = processFrames(frameCount)
    print("Found" + str(results) + " results!")