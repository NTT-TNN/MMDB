import numpy as np
import cv2
img = cv2.imread('/home/thao-nt/Desktop/CMS_Creative_164657191_Kingfisher.jpg')          # queryImage
np.savetxt('img1.txt', img, delimiter=" ", fmt="%s")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,img)
print (img)
np.savetxt('img2.txt', img, delimiter=" ", fmt="%s")
cv2.imwrite('CMS_Creative_164657191_Kingfisher.jpg',img)
