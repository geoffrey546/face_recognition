import cv2
#Loading the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#defining the function that will do the directions
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#we detect faces using this method
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)#detects the eys within the range of the face
        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    return frame
#doing some face recognizing using the webcam
video_capture = cv2.VideoCapture(0) #to capture video
while True:
    _ , frame = video_capture.read() #_ is used so as to receive only the second return value of video capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #COLOR_BGR2GRAY tells to do an average on blue green and red
    canvas = detect(gray, frame)
    cv2.imshow('Video', canvas) #will display all the functions
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
