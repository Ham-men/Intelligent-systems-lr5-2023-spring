import cv2
import numpy as np
import imutils



####### training part ###############
samples = np.loadtxt('generalsamples.data', np.float32)
responses = np.loadtxt('generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))
model = cv2.ml.KNearest_create()
model.train(samples, cv2.ml.ROW_SAMPLE, responses)
############################# testing part #########################
colors = {
    'yellow': (np.array([20, 180, 200]), np.array([40, 255,255]),np.array([10, 180, 200]), np.array([20, 255, 255])),
    'blue': (np.array([110, 160, 190]), np.array([130, 255,250]),np.array([90, 180, 190]), np.array([110, 255, 250])),
    'green': (np.array([45, 100, 50]), np.array([75, 255, 255])),
    'red': (np.array([170, 200, 150]), np.array([180, 255, 255]))
}
relatives = {
    0: ('Circle'),
    1: ('Hexagon'),
    2: ('Lightning'),
    3: ('Triangle'),
    4: ('Think'),
    5: ('Star'),
    6:('Rectangle')
}
image = cv2.imread('img4.jpg')
im = imutils. resize(image, width= 900 )
im2 = im.copy()

  #список фигур
figure_circle=0
figure_hexagon=0
figure_lightning=0
figure_triangle=0
figure_think=0
figure_star=0
figure_rectangle=0
count_fig=0  




def color_check():

   

    blurred = cv2.GaussianBlur(im, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv', hsv)
    for col in colors:
        mask = cv2.inRange(hsv, colors.get(col)[0],colors.get(col)[1])
        if col == 'green':
            cv2.imshow('test', mask)
        if col == 'blue':
            mask += cv2.inRange(hsv, colors.get(col)[2],colors.get(col)[3])
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = [i for i in contours if cv2.contourArea(i) > 50]

    find_fig(contours,mask,col,figure_circle,figure_hexagon,figure_lightning,figure_triangle,figure_think,figure_star,figure_rectangle,count_fig)
    #не хочет читать переменные которые обявленны в выше
    print('figure_circle = ',figure_circle)
    print('figure_hexagon = ',figure_hexagon)
    print('figure_lightning = ',figure_lightning)
    print('figure_triangle = ',figure_triangle)
    print('figure_think = ',figure_think)
    print('figure_star = ',figure_star)
    print('figure_rectangle = ',figure_rectangle)
    print('count_fig = ',count_fig)




def find_fig(contours,mask,col,figure_circle,figure_hexagon,figure_lightning,figure_triangle,figure_think,figure_star,figure_rectangle,count_fig):
     
    
    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            try:
                cv2.rectangle(im, (x, y), (x + w, y + h),(0, 255, 0), 2)
                roi = mask[y:y + h, x:x + w]
                l = float(w) / h
                roismall = cv2.resize(roi, (10, 10))
                roismall = roismall.reshape((1, 100))
                roismall = np.float32(roismall)
                retval, results, neigh_resp, dists = model.findNearest(roismall, k=1)
                num = int(results[0]) # ?
                result = relatives[num]
                print(num)
                count_fig+=1
                if(result=='Circle'):
                    figure_circle+=1
                if(result=='Hexagon'):
                    figure_hexagon+=1
                if(result=='Lightning'):
                    figure_lightning+=1
                if(result=='Triangle'):
                    figure_triangle+=1
                if(result=='Think'):
                    figure_think+=1
                if(result=='Star'):
                    figure_star+=1
                if(result=='Rectangle'):
                    figure_rectangle+=1

                text = "{} {}".format(col, result)
                cv2.putText(im2, text, (x + w // 2-80, y + h// 2), 0, 0.6, (0, 0, 0))
            except cv2.Error as e:
                print('Invalid')
 

    text2 = "1041 Смоляков Максим Анатольевич"
    cv2.putText(im2, text2, (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0))


color_check()


cv2.imshow('out', im2)
cv2.waitKey(0)
