import os
import sys
import time
import random
import math
import numpy as np
#import skimage.io
import imageio
import cv2
import tqdm
import pickle

import coco
import utils
import model as modellib
#import visualize

import xmltodict

def parse_scene_info(xml_path):
    assert os.path.isfile(xml_path), "{} Not found".format(xml_path)
    
    scene_data = dict()
    with open(xml_path) as fd:
        annotations = xmltodict.parse(fd.read())
    ## read weather: is here a typo? "sence_weather"(as shown in xml files) or it should be "sence_weather"?
    weather = annotations["sequence"]["sequence_attribute"]["@sence_weather"]
    scene_data["weather"] = weather
    
    ## read ignore region
    ignore_regions = list()
    if annotations["sequence"]["ignored_region"] != None:
        if isinstance(annotations["sequence"]["ignored_region"]["box"], dict):
            annotations["sequence"]["ignored_region"]["box"] = [annotations["sequence"]["ignored_region"]["box"]]
        for box in annotations["sequence"]["ignored_region"]["box"]:
            x1, y1, w, h = float(box["@left"]), float(box["@top"]), float(box["@width"]), float(box["@height"])
            x2, y2 = x1+w, y1+h
            ignore_regions.append([x1, y1, x2, y2])
    scene_data["ignore_region"] = ignore_regions
    
    return scene_data

def mask_to_contours(mask):
    _, contours, hierarchy = cv2.findContours(mask.copy(),
                                            cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                            offset=(0, 0))
    # return a list of contours
    return contours

def main():
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 8 # 8 is maximum for single P100.

    config = InferenceConfig()
    config.display()   
    
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)
    print("Successfully load coco pretrained model.")
    
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']
    
    
    # read scene
    info_dir = "../../detrac/DETRAC-Train-Annotations-XML/"
    video_dir = "../../detrac/Insight-MVT_Annotation_Train/"
    save_dir = "../../detrac/train_detect_output/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    videonames = [x for x in os.listdir(video_dir) if x.startswith("MVI")]
    print(videonames)
    batch_size = config.GPU_COUNT*config.IMAGES_PER_GPU
    for videoname in videonames:
        print("Processing scene {}...".format(videoname))
        video_files = sorted([img for img in os.listdir(os.path.join(video_dir, videoname)) if img.endswith(".jpg")])
        video_info = parse_scene_info(os.path.join(info_dir, videoname+".xml"))
        #vid = imageio.get_reader(video_file,  'ffmpeg')
        if video_info["weather"] == "night":
            print("Skipping {} due to night.".format(videoname))
            continue
        
        save_pkl_dir = os.path.join(save_dir, videoname)
        if not os.path.isdir(save_pkl_dir):
            os.makedirs(save_pkl_dir)
            start_point = 0 # set the start frame for detection
        else:
            start_point = len(os.listdir(save_pkl_dir))
        
        # tqdm progress bar
        pbar = tqdm.tqdm(total = len(video_files)-start_point)
        for fnum in range(start_point, len(video_files), batch_size):
            batch_data = []
            for i in range(batch_size):
                if fnum+i<len(video_files):
                    batch_data.append(cv2.imread(os.path.join(video_dir, videoname, video_files[fnum+i])))
                else:
                    batch_data.append(np.zeros((960,540,3)))
            #batch_data = np.array(batch_data)
            results = model.detect(batch_data, verbose=0)
            for i, r in enumerate(results):
                ## transform mask to contours
                print("frame {} detects {} objects".format(fnum+i, r['rois'].shape[0]))
                contours = []
                for m in r['masks'].transpose(2,0,1):
                    contours.append(mask_to_contours(m.astype('uint8')))
                r['contours'] = contours
                r.pop('masks', None) # This will return my_dict[key] if key exists in the dictionary, and None otherwise.
                ## save a single pkl for each frame
                if fnum + i < len(video_files):
                    with open(os.path.join(save_pkl_dir, video_files[fnum+i].replace(".jpg","")+".pkl"), "wb") as f:
                        pickle.dump(r, f, protocol=2)
                
                pbar.update(1)
        pbar.close()
                
    
if __name__ == "__main__":
    main()
