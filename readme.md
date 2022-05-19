## models framework from >> https://github.com/ultralytics/yolov5
## open-images custom data from >> https://storage.googleapis.com/openimages/web/index.html

# #

#### Steps

1) Edit 'trg_class' and 'user_path' in user_trg_config.yaml
2) Open custom_data.ipynb and run. Select newly created env as kernel. Data download would require between 8 to 16GB space. This would download images and labels, convert txt files into yolo format and transfer images into custom_data folder.
3) Open train_and_detect.ipynb. Kernel needs to be based on python 3.8. Run train.py with optional paramter specified in the code
4) Prepare test images and run detect.py with optional parameter specified in the code