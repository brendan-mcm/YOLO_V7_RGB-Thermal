# Program to show imgs w/ labels given main target directory
import cv2
import glob

main_tar_dir = "./final_data/th_video/train/"
imgs_dir = main_tar_dir + "images/"
labels_dir = main_tar_dir + "labels/"

list_labels = glob.glob(labels_dir+"*.txt")

rec_col = (255, 0, 0) # B of BGR
rec_thic = 2

label_col = rec_col
label_thic = 1
label_scale = 1


# needs specific class dict (1:1 vid set)
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


# helper function
# returns rectangle format 
# ((x1, y1) , (x2, y2)) in pixels
def yolo_rec (x, y, w, h, w_img, h_img):
    x_1 = (x - .5 * w) * w_img
    y_1 = (y - .5 * h) * h_img
    x_2 = (x + .5 * w) * w_img
    y_2 = (y + .5 * h) * h_img
    return ((int(x_1), int(y_1)), (int(x_2), int(y_2)))

# end helper functions

for label_f in list_labels:
    curr_img = cv2.imread(imgs_dir+label_f.split("/")[-1][:-4]+".jpg")
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