import cv2 as cv
import numpy as np

def pick_red_mask(src):
    hsv_img = cv.cvtColor(src, cv.COLOR_BGR2HSV)
    mean_v = np.mean(hsv_img[:, :, 2])

    if mean_v < 90:
        low1 = np.array([0, 30, 30])
        low2 = np.array([160, 30, 30])
    else:
        low1 = np.array([0, 60, 60])
        low2 = np.array([170, 60, 60])

    high1 = np.array([10, 255, 255])
    high2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv_img, low1, high1)
    mask2 = cv.inRange(hsv_img, low2, high2)
    combo = mask1 + mask2

    return cv.bitwise_and(src, src, mask=combo)

def find_octagons(bin_img, canvas):
    centers = []
    contours, _ = cv.findContours(bin_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    for c in contours:
        a = cv.contourArea(c)
        if a > 8000:
            p = cv.arcLength(c, True)
            approx = cv.approxPolyDP(c, 0.02 * p, True)

            if len(approx) == 8:
                x, y, w, h = cv.boundingRect(approx)
                cv.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv.putText(canvas, "Stop Sign", (x + w + 20, y + 20),
                           cv.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0,255), 2)
                centers.append([x + w // 2, y + h // 2])
    return centers

def draw_stop_signs(frame, collected):
    out_img = frame.copy()

    red_only = pick_red_mask(frame)
    blurred = cv.GaussianBlur(red_only, (7, 7), 2)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

    k = np.ones((7, 7))
    edges = cv.Canny(gray, 30, 40)
    thick = cv.dilate(edges, k, iterations=1)

    pts = find_octagons(thick, out_img)
    collected.extend(pts)

    out_img = cv.resize(out_img, (960, 720))
    return out_img

sign_centers = []

img1 = cv.imread("c:/Users/ASUS/Desktop/stop_sign_dataset/stop_sign1.jpg")
img2 = cv.imread("c:/Users/ASUS/Desktop/stop_sign_dataset/stop_sign2.jpg")
img3 = cv.imread("c:/Users/ASUS/Desktop/stop_sign_dataset/stop_sign3.jpg")
img4 = cv.imread("c:/Users/ASUS/Desktop/stop_sign_dataset/stop_sign4.jpg")
img5 = cv.imread("c:/Users/ASUS/Desktop/stop_sign_dataset/stop_sign5.jpg")

result1 = draw_stop_signs(img1, sign_centers)
result2 = draw_stop_signs(img2, sign_centers)
result3 = draw_stop_signs(img3, sign_centers)
result4 = draw_stop_signs(img4, sign_centers)
result5 = draw_stop_signs(img5, sign_centers)

stack = [result1, result2, result3, result4, result5]

for i in range(len(stack)):
    cv.imshow(f"Stop Sign {i + 1}", stack[i])
    print(f"Stop sign coordinates of the {i + 1}. image: {sign_centers[i]}")

key = cv.waitKey(0)