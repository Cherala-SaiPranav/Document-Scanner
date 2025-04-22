import cv2
import numpy as np
from imutils.perspective import four_point_transform
import pytesseract

cap = cv2.VideoCapture(0 + cv2.CAP_V4L2)

img_count = 0
txt_count = 0
scale = 0.5

font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

width, height = 800,800
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def image_processing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return threshold

def scan_detection(image):
    global document_contour

    document_contour = np.array([[0,0], [width, 0], [width, height], [0, height]])

    # converting image BGR to grey
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Reducing noise with Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # turning the image into pure black and white (BINARY) using Otsu's method
    _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # finding binary image in contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # sorting the detected contours from largest to smallest by area(size)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)\
    
    max_area = 0
    for contour in contours:

        # checking if the area of contour is greater than 1000
        area = cv2.contourArea(contour)
        if area > 1000:

            # calculate the perimeter of contour
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * perimeter, True)

            # checking if the contour has 4 points and is the biggest so far
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area

    
    cv2.drawContours(frame, [document_contour], -1, (0, 255, 0), 3)
    
def center_txt(image, text):
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    text_x = (image.shape[1] - text_size[0]) // 2
    text_y = (image.shape[0] - text_size[1]) // 2
    cv2.putText(image, text, (text_x, text_y), font, 2, (255, 0, 255), 5, cv2.LINE_AA)

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame")
        break

    frame_copy = frame.copy()

    scan_detection(frame_copy)

    cv2.imshow("input", cv2.resize(frame, (int(scale * width), int(scale * height))))

    # wraping the document in frame
    warped = four_point_transform(frame_copy, document_contour.reshape(4, 2))
    # cv2.imshow("Warped", cv2.resize(warped, (int(scale * width), int(scale * height))))

    processed = image_processing(warped)
    if processed.shape[0] > 20 and processed.shape[1] > 20:
        processed = processed[10:processed.shape[0] - 10, 10:processed.shape[1] - 10]
        # cv2.imshow("Processed", cv2.resize(processed, (int(scale * width), int(scale * height))))
    else:
        print("Processed image too small to crop.")



    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == 27:
        break

    elif key_pressed == ord('s'):
        cv2.imwrite("doc_output/Document_" + str(img_count) + ".jpg", processed)
        img_count += 1

        center_txt(frame, "Scan Saved")
        cv2.imshow("input", cv2.resize(frame, (int(scale * width), int(scale * height))))
        cv2.waitKey(750)

    elif key_pressed == ord('o'):
        file = open("txt_output/Doc_text_" + str(txt_count) + ".txt", "w")
        txt_count += 1

    # OCR (Optical Character Recognition) extracting text from warped document image
        ocr_text = pytesseract.image_to_string(warped)
        file.write(ocr_text)
        file.close()

        center_txt(frame, "text Saved")
        cv2.imshow("imput", cv2.resize(frame, (int(scale * width), int(scale * height))))
        cv2.waitKey(750)

cap.release()
cv2.destroyAllWindows()