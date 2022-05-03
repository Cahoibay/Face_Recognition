import cv2
from simple_facerec import SimpleFacerec

#Trích xuất đặc trưng từ ảnh khuôn mặt trogn folder images
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

#Bật camera
cap = cv2.VideoCapture(0)

while True :
    #Get Frame
    ret,frame = cap.read()
    #Xác định khuôn mặt chứa trong khung hình chụp đc từ webcam
    face_locations,face_names = sfr.detect_known_faces(frame)
    for face_loc,name in zip(face_locations,face_names) :
        y1 ,x1,y2,x2= face_loc[0],face_loc[1],face_loc[2],face_loc[3]
        #Gắn nhãn tên và tạo khung chứa khuôn mặt
        cv2.putText(frame,name,(x2,y2+30),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,127),2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,204,229),6)

    cv2.imshow("Frame",frame)

    if cv2.waitKey(50) == ord("q"):
        break

#cap.realease()
cv2.destroyAllWindows()

