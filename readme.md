## models framework from >> https://github.com/ultralytics/yolov5
## open-images custom data from >> https://storage.googleapis.com/openimages/web/index.html

# #

#### Steps
1) Edit 'trg_class' and 'user_path' in custom_data.py and run. This would download images and labels, convert txt files into yolo format and transfer images into custom_data folder.
2) run train.py with optional paramter specified in yolov5 repo
3) use the best.pt weight of your preferred run iteration as a paramater to parse when running detect.py to either test image set in custom_data or your own test image set