import cv2
import imutils

path = "../resource/discuz/ndl4.jpg"

image = cv2.imread(path)
# cv2.imshow("image",image)
# cv2.waitKey(0)

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
# cv2.imshow("gray",gray)
# cv2.waitKey(0)

threhold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(threhold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if imutils.is_cv2() else cnts[1]

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

for c in cnts:
    (x, y, w, h) = cv2.boundingRect(c)
    roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]
    cv2.imshow("ROI", imutils.resize(roi, width=28))
    key = cv2.waitKey(0)

cv2.imshow("threhold", threhold)
cv2.waitKey(0)



