import cv2
import glob
import json
import numpy

rgb_img_dir = "./channel_testing/video-BzZspxAweF8AnKhWK-frame-000757-Za9ZJDCPR8q2Miism.jpg"
th_img_dir = "./channel_testing/video-4FRnNpmSmwktFJKjg-frame-000757-4Wp8vfKimH7YBrgof.jpg"

rgb_img = cv2.imread(rgb_img_dir)
th_img = cv2.imread(th_img_dir)

h_t, w_t, _ = th_img.shape
new_dim = (w_t, h_t)

# resize RGB to match thermal
print("Original dimensions : ", rgb_img.shape)

resized_rgb = cv2.resize(rgb_img, new_dim) # other ways for interpolation
print("new dimensions = ", resized_rgb.shape)

print("Thermal dimensions = ", th_img.shape)

# seperate channels
b, g, r = cv2.split(resized_rgb)

colorized_t = cv2.applyColorMap(th_img, cv2.COLORMAP_PLASMA)
cv2.imshow("colorized", colorized_t)
b_t, g_t, r_t = cv2.split(colorized_t)

merged_img = cv2.merge([b, g, r, b_t, g_t, r_t])

cv2.imshow("thermal", th_img)
cv2.imshow("resized", resized_rgb)

cv2.imwrite("colorized.jpg", colorized_t)
# cv2.imshow("merged", merged_img)

# deconstruct merge and recreate
print("num py = ", numpy.shape(merged_img))
reshape_merge = numpy.dsplit(merged_img, 2)
print("reshape = ", numpy.shape(reshape_merge))

print("arr[0] = ",reshape_merge[0])
print("arr[1] = ",reshape_merge[1])

# re_rgb = cv2.merge([r_m, g_m, b_m])
# re_th = cv2.merge([r_tm, g_tm, b_tm])

cv2.imshow("re rgb", reshape_merge[0])
cv2.imshow("re th", reshape_merge[1])

cv2.waitKey(0)
cv2.destroyAllWindows()

