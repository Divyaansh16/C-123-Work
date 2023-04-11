import math
import cv2
import mediapipe as mp
from pynput.mouse import Button,Controller
import pyautogui
mymouse=Controller()
pinch=False
video=cv2.VideoCapture(0)
width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width,height)
hands=mp.solutions.hands
drawing=mp.solutions.drawing_utils
hand_obj=hands.Hands(min_detection_confidence=0.75,
                     min_tracking_confidence=0.75)
(screen_width,screen_height)=pyautogui.size()
print(screen_width,screen_height)
def countFingures(lst,myimage):
    count=0
    global pinch
    thresh=(lst.landmark[0].y*100-lst.landmark[9].y*100)/2
    #print("What is the thresh value:",thresh)
    if(lst.landmark[5].y*100-lst.landmark[8].y*100)>thresh:
        count+=1
    if (lst.landmark[9].y * 100 - lst.landmark[12].y * 100) > thresh:
        count += 1
    if (lst.landmark[13].y * 100 - lst.landmark[16].y * 100) > thresh:
        count += 1
    if (lst.landmark[17].y * 100 - lst.landmark[20].y * 100) > thresh:
        count += 1
    if (lst.landmark[5].x * 100 - lst.landmark[4].x * 100) >6:
        count += 1
    totalfingure=count
    fingure_tip_x=int(lst.landmark[8].x*width)
    fingure_tip_y=int(lst.landmark[8].y*height)
    thumb_tip_x=int(lst.landmark[4].x*width)
    thumb_tip_y=int(lst.landmark[4].y*height)
    cv2.line(myimage,(fingure_tip_x,fingure_tip_y),(thumb_tip_x,thumb_tip_y),(255,0,0),2)
    center_x=int((fingure_tip_x+thumb_tip_x)/2)
    center_y=int((fingure_tip_y+thumb_tip_y)/2)
    cv2.circle(myimage,(center_x,center_y),2,(0,0,255),2)
    distace=math.sqrt(((fingure_tip_x-thumb_tip_x)**2)+((fingure_tip_y-thumb_tip_y)**2))
    #print("What is distace:",distace)
    relative_mouse_x=(center_x/width)*screen_width
    relative_mouse_y=(center_y/height)*screen_height
    mymouse.position=(relative_mouse_x,relative_mouse_y)
    print("What is mouse position:",mymouse.position)
    if distace>40:
        if pinch==True:
            pinch=False
            mymouse.release(Button.left)
    if distace<=40:
        if pinch==False:
            pinch=True
            mymouse.press(Button.left)
    return totalfingure
while True:
    dummy,image=video.read()
    flipimage=cv2.flip(image,1)
    result=hand_obj.process(cv2.cvtColor(flipimage,cv2.COLOR_BGR2RGB))
    if result.multi_hand_landmarks:
        hand_keyPoints=result.multi_hand_landmarks[0]
        #print(hand_keyPoints)
        count=countFingures(hand_keyPoints,flipimage)
        #print("What is fingure count:",count)
        cv2.putText(flipimage, "Fingures " + str(count), (200, 100), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(flipimage,"pinch "+str(pinch),(100,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
        drawing.draw_landmarks(flipimage,hand_keyPoints,hands.HAND_CONNECTIONS)
    cv2.imshow("Hand Gestures:",flipimage)
    key=cv2.waitKey(1)
    if key==27:
        break
video.release()
cv2.destroyAllWindows()