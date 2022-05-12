import os
import shutil
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import yaml

### images download

## specify target class to download
trg_class = "Box"

## scecify user path
user_path = "/Users/don.pa"

DATA_PATH = os.path.join(user_path,f"fiftyone/open-images-v6")

## now we're suppposed to be under project folder
main_path = os.getcwd()
OUTPUT_PATH = os.path.join(main_path, f"custom_data")

## define data processing function
def process_data(data, data_type="train"):
    for _, row in tqdm(data.iterrows(), total=len(data)):
        image_name = row["image_id"]
        bboxes = row["bboxes"]
        # label = row['class_id']
        label = 0
        yolo_data = []
        for bbox in bboxes:
            x_center = round((bbox[1]+bbox[0])/2,6)
            y_center = round((bbox[3]+bbox[2])/2,6)
            w = round(bbox[1]-bbox[0],6)
            h = round(bbox[3]-bbox[2],6)

            yolo_data.append([label, x_center, y_center, w, h])
        yolo_data = np.array(yolo_data)

        np.savetxt(
            os.path.join(OUTPUT_PATH, f"labels/{data_type}/{image_name}.txt"),
            yolo_data,
            fmt=["%d","%f","%f","%f","%f"]
        )
        shutil.copyfile(
            os.path.join(DATA_PATH, f"{data_type}/data/{image_name}.jpg"),
            os.path.join(OUTPUT_PATH, f"images/{data_type}/{image_name}.jpg")
        )

if __name__ == "__main__":

    ## download images

    dataset = foz.load_zoo_dataset(
        "open-images-v6",
        split=['train','validation'],
        label_types="detections",
        classes=[trg_class]
        # ,
        # max_samples=500,
    )

    ### create custom data folder with below tree structure

    # custom_data
    #   images
    #       train
    #       validation
    #   labels
    #       train
    #       validation

    custom_data_path = os.path.join(main_path,f"custom_data")
    path_list = [custom_data_path]
    for fldr in ['images','labels']:
        for data_type in ['train','validation']:
            path = os.path.join(custom_data_path,f"{fldr}/{data_type}")
            path_list.append(path)

    for path in path_list:
        if path == custom_data_path: 
            ## remove custom_data path if exist
            if os.path.exists(path):
                shutil.rmtree(path)
                os.mkdir(path)
        else:
            if not os.path.exists(path):
                os.mkdir(path)


    ## get class names
    class_file = os.path.join(DATA_PATH, f"train/metadata/classes.csv")
    class_dict= pd.read_csv(class_file, names=['path','class']).reset_index() \
    .rename(columns={'index':'class_id'}).set_index('path').to_dict('index')
    # class_dict

    for data_type in ["train","validation"]:

        detection_file = os.path.join(DATA_PATH, 
                                     f"{data_type}/labels/detections.csv")
        ## get relevant image ids
        imagefile_path = os.path.join(DATA_PATH, 
                                     f"{data_type}/data")
        image_ids = [re.split('\.',f )[0] for f in os.listdir(imagefile_path) if os.path.isfile(os.path.join(imagefile_path, f))]

        df_detects= pd.read_csv(detection_file)
        df_detects = df_detects.rename(columns={'ImageID':'image_id'})
        imgid_filter = df_detects['image_id'].isin(image_ids)
        df_rel_detects = df_detects.iloc[np.where(imgid_filter)].reset_index(drop=True)
        df_rel_detects['class_id'] = df_rel_detects['LabelName'].apply(lambda x: None if x not in class_dict.keys() else class_dict[x]['class_id'])
        df_rel_detects['class'] = df_rel_detects['LabelName'].apply(lambda x: None if x not in class_dict.keys() else class_dict[x]['class'])
        class_filter = df_rel_detects['class'].str.lower().isin([i.lower() for i in trg_class])
        df_rel_detects = df_rel_detects.iloc[np.where(class_filter)].reset_index(drop=True)

        df_rel_detects['bboxes'] = df_rel_detects.apply(lambda row: [row['XMin'],row['XMax'],row['YMin'],row['YMax']],axis=1)
        df_rel_detects = df_rel_detects.groupby(by=['image_id','class_id'])['bboxes'].apply(list).reset_index()
        # print(df_rel_detects.shape)
        
        ### moving images and labels to custom_data path
        ### edit labels to fit yolo format
        process_data(df_rel_detects,data_type=data_type)

        ## dump dict object into yaml file
        custom = {
            'train': "custom_data/images/train",
            'val': "custom_data/images/validation",
            'nc': 1,
            'names': [trg_class]
        }

        with open('custom.yaml', 'w') as f:
            data = yaml.dump(custom, f)