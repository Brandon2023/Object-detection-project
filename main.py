import cv2 as cv

cam = cv.VideoCapture(0)
package_data = cv.CascadeClassifier('haarcascade_eye.xml.xml')

if not cam.isOpened():
    print("Camera not working")
    exit()

def package(frame):
    frame_colour = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    box = package_data.detectMultiScale(frame_colour)
    for (x,y,w,h) in box:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        print("eyes detected")

while True:
    ret, frame = cam.read()
    package(frame)

    if ret is None:
        print("no stream") 
        break

    cv.imshow('package detector', frame)
    if cv.waitKey(1) == ord('w'):
        break

cam.release()
cv.destroyAllWindows()
