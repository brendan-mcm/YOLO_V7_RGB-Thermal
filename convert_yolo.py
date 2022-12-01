import cv2
import glob
import json

# Different for train and test
tar_dir = "./mini_flir/test/"

list_imgs = glob.glob(tar_dir + "images/*.jpg")
labels_tar = tar_dir+"labels/"
json_tar = tar_dir+"index.json"

json_f = open(json_tar)
label_inf = json.load(json_f)
json_f.close()


# with additions of rider, license plate, face
rgb_class_dict = {
    "person": 0,
    "bike": 1,
    "car": 2,
    "motor": 3,
    "bus": 4,
    "train": 5,
    "truck": 6,
    "light": 7,
    "hydrant": 8,
    "sign": 9,
    "skateboard": 10,
    "stroller": 11,
    "scooter": 12,
    "other vehicle": 13,
    "rider": 14,
    "license plate": 15,
    "face": 16
}

inv_rgb_class_dict = {v: k for k, v in rgb_class_dict.items()}

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

for img_name in list_imgs:
    img_title = img_name.split("/")[-1]
    img_splt = img_title.split("-")[-1].split(".")[0]
    # print(img_title)
    
    for frame in label_inf['frames']:
        if frame['datasetFrameId'] == img_splt:
            # print("Match")
            print(img_splt)
            curr_f = open(labels_tar+img_title[:-4]+".txt", 'w')
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
                print(combined_str)
                curr_f.write(combined_str)
            curr_f.close()
            break


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