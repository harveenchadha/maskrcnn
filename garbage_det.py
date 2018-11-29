"""

Custom RCNN code for Training your own dataset

Changes to try on our own datset

create a folder with the train and val images in seperate sub folders
use VGG Image Annotator to annotate the data's and save the json file into the subfolders

add maping of interger to our class name dct = {"nine":1,"ten":2,"jack":3,"queen":4,"king":5,"Ace":6}
make a list of class names class_names=['nine','ten','jack','queen','king','ace']

change NAME = x choose any name

change NUM_CLASSES = 1 + i  # Background + toy where i is the total classes

Add th below with the name choose as x
self.add_class(x, 1, "nine")
self.add_class(x, 2, "ten")
self.add_class(x,3,"jack")
self.add_class(x,4,"queen")
self.add_class(x,5,"king")
self.add_class(x,6,"Ace")

self.add_image(x,...)

if info["source"] != x:

and execute the below commands to get results 

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    python3 garbage_det.py train --dataset=/garbageImages/ --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    	 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
    python3 garbage_det.py detect --weights='/mask_rcnn_garbage_0010.h5' --image='14.jpg'

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

#label variables
dct = {"garbage":1}
class_names=['garbage']

############################################################
#  Configurations
############################################################


class CustomConfig(Config):	
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "garbage"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + toy

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.5


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        self.add_class("garbage", 1, "garbage")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = ROOT_DIR+'/'+dataset_dir
        dataset_dir = os.path.join(dataset_dir, subset)
        print(dataset_dir)

        annotations = json.load(open(os.path.join(dataset_dir, "garbage.json")))
        annotations = list(annotations.values())  

        annotations = [a for a in annotations if a['regions']]


        # Add images
        for a in annotations:	
            
            #The polygon coordinates and the label name
            polygons = [r['shape_attributes'] for r in a['regions']]
            objects = [s['region_attributes'] for s in a['regions']]
            
            #The label name into corresponding interger
            num_ids = [int(dct[n['class name']]) for n in objects]

            #Loading the image and getting the width and hight
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            
            #Add the image to object with all details
            self.add_image(
                "garbage",  
                image_id=a['filename'], 
                path=image_path,
                width=width, height=height,
                polygons=polygons,num_ids=num_ids)

    def load_mask(self, image_id):

        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]

        if info["source"] != "garbage":
            return super(self.__class__, self).load_mask(image_id)

        #Getting the label and initalizing the mask
        num_ids = info['num_ids']        
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        #looping through the mask to apply them
        for i, p in enumerate(info["polygons"]):            
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            

        #Array of labels with their id
        num_ids = np.array(num_ids, dtype=np.int32)

        return mask.astype(np.bool), num_ids	


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cards":
            return info["path"]	
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    
    #Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    #Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    #Model is started to train after loading the datasets
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=10,
                layers='heads')

#Function to plot in axes
def get_ax(rows=1, cols=1, size=16):

    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

import os

#Detection on new image
def detect_and_save(model1, image_path=None, video_path=None):
    
    assert image_path or video_path

    if image_path:
#         image = skimage.io.imread(args.image)
#         r = model.detect([image], verbose=1)[0]
        
        #loading the image and detecting
        image = skimage.io.imread(image_path)
        class InferenceConfig(CustomConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
        config.display()
        model = modellib.MaskRCNN(mode="inference", config =config, model_dir = DEFAULT_LOGS_DIR)
        print(model)
        model.load_weights('./mask_rcnn_garbage_0010.h5', by_name=True)
        r = model.detect([image], verbose=1)[0]
        

#         ax = get_ax(1)
        final_image=display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            class_names, r['scores'], 
                            title="Predictions")	
        
        #save the output
        file_name = "prediction_{:%Y%m%dT%H%M}.jpg".format(datetime.datetime.now())
        score = r['scores']
        skimage.io.imsave(file_name, final_image)
        if(len(score)!=0 ):
          if(score[0] < 0.5):
            return False
          else:
            return True
        else:
          return False
        

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "predction_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

 
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "detect":
        detect_and_save(model, image_path=args.image,
                                video_path=args.video)
        #return ret
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect	'".format(args.command))
