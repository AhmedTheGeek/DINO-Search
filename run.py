import argparse
import os
import sys

import numpy as np
import cv2
import gradio as gr

from groundingdino.util.inference import load_model, load_image, predict, annotate
import groundingdino.datasets.transforms as T
from PIL import Image

def image_transform_grounding(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return init_image, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

def getFrame(vidcap, sec = 0, count = 0, frameRate = 0, images = list()):
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap.read()
    if hasFrames:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _, image_tensor = image_transform_grounding(Image.fromarray(color_coverted))
        images.append((image, image_tensor))
        print("\Extracted %i frames" % count, end="\r")
        return getFrame(vidcap, sec + frameRate, count + 1, frameRate, images)
    return images

def progress_bar(current, total, bar_length=20, label = 'Processing frames'):
    fraction = current / total

    arrow = int(fraction * bar_length - 1) * '-' + '>'
    padding = int(bar_length - len(arrow)) * ' '

    ending = '\n' if current == total else '\r'

    print(f'{label}: [{arrow}{padding}] {int(fraction*100)}%', end=ending)

    
def processFrames(images, keywords, box_threshold, text_threshold):
    processed = 5
    resultsFound = 0
    results = list()
    while processed < len(images):
        # run model
        # load image
        init_image, image_tensor  = images[processed]
        boxes, logits, phrases = predict(
            model, image_tensor, keywords, box_threshold, text_threshold
        )
        
        image_pil: Image = image_transform_grounding_for_vis(Image.fromarray(init_image))

        if len(boxes) > 0:   
            annotated_frame = annotate(image_source=np.asarray(image_pil), boxes=boxes, logits=logits, phrases=phrases)
            # cv2.imwrite("./output/results/annotated_image"+str(processed)+".jpg", annotated_frame)
            image_with_box = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
            results.append(image_with_box)
            resultsFound += len(boxes)
            
        processed += 1
        progress_bar(processed, len(images))
    
    return results


def app(video, keywords, extraction_framerate, box_threshold, text_threshold):
    # # Extract video frames
    vidcap = cv2.VideoCapture(video)
    
    print("Extracting video frames")
    
    images = getFrame(vidcap, 0, 0, frameRate=extraction_framerate)

    results = processFrames(images, keywords, box_threshold, text_threshold)
    
    print("Found " + str(len(results)) + " results!")
    
    return results

if __name__ == "__main__":
    # parser = argparse.ArgumentParser("Grounding DINO Search Engine", add_help=True)
    # parser.add_argument("--input_video", "-i", type=str, required=True, help="path to video file")
    # parser.add_argument("--keywords", "-k", type=str, required=True, help="keywords to find comma seperated")
    # parser.add_argument("--output", "-o", type=str, required=True, help="output path")
    # parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    # parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    # parser.add_argument("--framerate", type=float, default=0.5, help="frame extraction rate in seconds")
    # parser.add_argument("--cpu-only", action="store_true", help="running on cpu only!, default=False")

    # args = parser.parse_args()

    # video = args.input_video
    # keywords = args.keywords
    # box_threshold = args.box_threshold
    # text_threshold = args.text_threshold
    # framerate = args.framerate
    # output = args.output
    
    model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")
        
    block = gr.Blocks().queue()
    with block:
        gr.Markdown("# [DINO Search](https://github.com/AhmedTheGeek/DINO-Search)")

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(source='upload')
                grounding_caption = gr.Textbox(label="Search Keywords")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    extraction_framerate = gr.Slider(
                        label="Extraction Framerate", minimum=0.0, maximum=1.0, value=0.5, step=0.1
                    )
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )

            with gr.Column():
                # gallery = gr.outputs.Gallery().style(full_width=True, full_height=True)
                gallery = gr.Gallery(label="Generated images", show_label=False).style(
                        grid=[1], height="auto", container=True, full_width=True, full_height=True)

        run_button.click(fn=app, inputs=[
                        input_video, grounding_caption, extraction_framerate, box_threshold, text_threshold], outputs=gallery)  
        
    block.launch(server_name='0.0.0.0', server_port=7579)
