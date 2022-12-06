import cv2
import glob
import json
import random

og_rgb_vid_dir = "./main_data/FLIR_ADAS_v2/video_rgb_test/data/"
og_rgb_json_f = "./main_data/FLIR_ADAS_v2/video_rgb_test/index.json"

og_th_vid_dir = "./main_data/FLIR_ADAS_v2/video_thermal_test/data/"
og_th_json_f = "./main_data/FLIR_ADAS_v2/video_thermal_test/index.json"

# List of all RGB images
og_rgb_list_imgs = glob.glob(og_rgb_vid_dir+ "*.jpg")

# Open and save RGB -> TH json
one_json_f = open("./main_data/rgb_to_thermal_vid_map.json")
one_one = json.load(one_json_f)
one_json_f.close()

# Open and save RGB json labels
rgb_json_f = open(og_rgb_json_f)
rgb_labels = json.load(rgb_json_f)
rgb_json_f.close()

# Open and save TH json labels
th_json_f = open(og_th_json_f)
th_labels = json.load(th_json_f)
th_json_f.close()


# Different for video RGB/thermal test (1:1). 12 classes total.
rgb_class_dict = {
    "person": 0,
    "bike": 1,
    "car": 2,
    "motor": 3,
    "truck": 4,
    "light": 5,
    "hydrant": 6,
    "sign": 7,
    "other vehicle": 8,
    "rider": 9,
    "face": 10,
    "dog": 11,
    "license plate": 12
}
inv_rgb_class_dict = {v: k for k, v in rgb_class_dict.items()}


# Begin helper functions

# returns coordinates in tuple form, given top left with width and height. 
# format is (x', y', w', h')
def xywh_yolo (x, y, w_b, h_b, w_img, h_img):
    x_mod = (x + .5 * w_b) / w_img
    y_mod = (y + .5 * h_b) / h_img
    w_mod = w_b / w_img
    h_mod = h_b / h_img
    return (x_mod, y_mod, w_mod, h_mod)

# returns rectangle format 
# ((x1, y1) , (x2, y2)) in pixels
def yolo_rec (x, y, w, h, w_img, h_img):
    x_1 = (x - .5 * w) * w_img
    y_1 = (y - .5 * h) * h_img
    x_2 = (x + .5 * w) * w_img
    y_2 = (y + .5 * h) * h_img
    return ((int(x_1), int(y_1)), (int(x_2), int(y_2)))

# End helper functions


# Shuffle / split data into train (70%), validate (15%), test (15%)
# Shuffle (not accounting for class imbalancing at this stage)
random.shuffle(og_rgb_list_imgs)

train_cutoff = int(len(og_rgb_list_imgs)*.7) # roughly 70%
val_cutoff = int(len(og_rgb_list_imgs)*.15) + train_cutoff # roughly 15% each

train_imgs = og_rgb_list_imgs[:train_cutoff]
val_imgs = og_rgb_list_imgs[train_cutoff:val_cutoff]
test_imgs = og_rgb_list_imgs[val_cutoff:]

imgs_lists = []
imgs_lists.append(train_imgs)
imgs_lists.append(val_imgs)
imgs_lists.append(test_imgs)

for i in range(3): # 0 = train, 1 = val, 2 = test
    curr_list = imgs_lists[i]
    for rgb_img_name in curr_list: 
        
        rgb_title = rgb_img_name.split("/")[-1]
        rgb_splt = rgb_title.split("-")[-1].split(".")[0]
        
        if rgb_title not in one_one: # check if RGB has 1:1 correspondence
            continue

        th_title = one_one[rgb_title]
        # print(rgb_title +" matches with "+th_title) # worked

        rgb_save_dir = "./"
        th_save_dir = "./"
        
        if i == 0:
            rgb_save_dir = "./final_data/rgb_video/train/"
            th_save_dir = "./final_data/th_video/train/"
        elif i == 1:
            rgb_save_dir = "./final_data/rgb_video/val/"
            th_save_dir = "./final_data/th_video/val/"
        else:
            rgb_save_dir = "./final_data/rgb_video/test/"
            th_save_dir = "./final_data/th_video/test/"

        # Write and save RGB label
        for frame in rgb_labels['frames']:
            if frame['datasetFrameId'] == rgb_splt:
                
                curr_f = open(rgb_save_dir+ "labels/rgb_"+rgb_splt+".txt", 'w')
                w_img = frame['width']
                h_img = frame['height']
                for annote in frame['annotations']:
                    label = annote['labels'][0]
                    box = annote['boundingBox']
                    x = box['x']
                    y = box['y']
                    w = box['w']
                    h = box['h']

                    yolo_form = xywh_yolo(x, y, w, h, w_img, h_img)
                    combined_str = str(rgb_class_dict[label])+ " " + str(yolo_form[0]) + " " + str(yolo_form[1]) + " " + str(yolo_form[2]) + " " + str(yolo_form[3]) +"\n"
                    curr_f.write(combined_str)
                curr_f.close()
                break
        
        # Copy RGB img file over
        rgb_img_read = cv2.imread(og_rgb_vid_dir+rgb_title)
        cv2.imwrite(rgb_save_dir+"images/rgb_"+rgb_splt+".jpg", rgb_img_read)

        # Write and save TH label
        th_splt = th_title.split("-")[-1].split(".")[0]
        for frame in th_labels['frames']:
            if frame['datasetFrameId'] == th_splt:
                
                curr_f = open(th_save_dir+ "labels/th_"+rgb_splt+".txt", 'w') # note kept same ID
                w_img = frame['width']
                h_img = frame['height']
                for annote in frame['annotations']:
                    label = annote['labels'][0]
                    box = annote['boundingBox']
                    x = box['x']
                    y = box['y']
                    w = box['w']
                    h = box['h']

                    yolo_form = xywh_yolo(x, y, w, h, w_img, h_img)
                    combined_str = str(rgb_class_dict[label])+ " " + str(yolo_form[0]) + " " + str(yolo_form[1]) + " " + str(yolo_form[2]) + " " + str(yolo_form[3]) +"\n"
                    curr_f.write(combined_str)
                curr_f.close()
                break
        
        # Copy Th img file over
        th_img_read = cv2.imread(og_th_vid_dir+th_title)
        colorized_th = cv2.applyColorMap(th_img_read, cv2.COLORMAP_PLASMA)
        cv2.imwrite(th_save_dir+"images/th_"+rgb_splt+".jpg", colorized_th) # note kept same ID


'''


# Reconstruction
list_labels = glob.glob(labels_tar+"*.txt")
# print(list_labels)

rec_col = (255, 0, 0) # BGR
rec_thic = 2

label_col = rec_col
label_thic = 3
label_scale = .9


for label_f in list_labels:
    curr_img = cv2.imread(tar_dir+"images/"+label_f.split("/")[-1][:-4]+".jpg")
    h, w, _ = curr_img.shape
    
    curr_f = open(label_f, 'r')
    for line in curr_f.readlines():
        attrs = line.split() # along whitespace by default
        rec = yolo_rec(float(attrs[1]), float(attrs[2]), float(attrs[3]), float(attrs[4]), w, h)
        class_label = inv_rgb_class_dict[int(attrs[0])]
        print(rec)
        curr_img = cv2.rectangle(curr_img, rec[0], rec[1], rec_col, rec_thic)
        curr_img = cv2.putText(curr_img, class_label, rec[0], cv2.FONT_HERSHEY_PLAIN, label_scale, label_col, label_thic)
    
    cv2.imshow(label_f, curr_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




'''
'''
recs_list = []
for frame in label_inf["frames"]:
    for annote in frame['annotations']:
        label = annote["labels"][0] # multiple ever? don't think so
        box = annote["boundingBox"]
        x = box['x']
        y = box['y']
        w = box['w']
        h = box['h']
        print("x = "+ str(x) + " y = "+ str(y) +" w = "+ str(w)+ " h = "+ str(h))
        print(box)
        recs_list.append( ((x, y), (x + w, y + h), label) )


img = cv2.imread(imgpath)

window_name = 'test_img'

# rectangle (FLIR format is x,y top left corner of box then width and height
top_l = (643 , 578 ) # (x), (y)
bot_r = (643 + 70, 578 + 15) # (x + w), (y + h)

# BGR
rec_col = (255, 0, 0)
# 2 px
thickness = 2

curr_img = img
for rec in recs_list:
    curr_img = cv2.rectangle(curr_img, rec[0], rec[1], rec_col, thickness)
    curr_img = cv2.putText(curr_img, rec[2], rec[0], cv2.FONT_HERSHEY_PLAIN, 0.9, rec_col, thickness)

cv2.imshow(window_name, curr_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''