import hydra
from omegaconf import DictConfig
import cv2
import numpy as np           
import os
import matplotlib.pyplot as plt


def show(img,title,pause=2):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Отображение через matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"{title}")
    plt.axis('off')
    plt.draw()
    plt.pause(pause)  # Короткая пауза для отрисовки


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    
    if os.path.isdir(f"{cfg.results_folder}"):
        print(f"Каталог {cfg.results_folder} существует")
    else:
        print(f"Каталог {cfg.results_folder} не существует")
        exit(0)
        
    # Reading the image by parsing the argument
    
    if os.path.isfile(cfg.img_filename):
        print(f"Image FileName: {cfg.img_filename} is exist")
    else:
        print(f"Image FileName: {cfg.img_filename} not exist")
        exit(0)
        
    img = cv2.imread(cfg.img_filename)
    img = cv2.resize(img ,((int)(img.shape[1]/5),(int)(img.shape[0]/5)))
    original = img.copy()
    neworiginal = img.copy()
    
    cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_1.jpg"), img)
    show(img,'original step 1')
    

    # Calculating number of pixels with shade of white(p) to check if exclusion of these pixels is required or not 
    # (if more than a fixed %) in order to differentiate the white background or white patches in image 
    # caused by flash, if present.
    # Расчет количества пикселей с оттенком белого (p) для проверки необходимости исключения этих пикселей 
    # (если их больше фиксированного %) с целью дифференциации белого фона или белых пятен на изображении, 
    # вызванных вспышкой, если они присутствуют.
    
    p = 0 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            if (B > 110 and G > 110 and R > 110):
                p += 1
    #finding the % of pixels in shade of white
    # поиск % пикселей в оттенке белого
    totalpixels = img.shape[0]*img.shape[1]
    per_white = 100 * p/totalpixels
    
    print(f'percantage of white: {per_white}\ntotal: {totalpixels}\nwhite: {p}\n')
    
    # excluding all the pixels with colour close to white if they are more than 10% in the image
    # исключая все пиксели с цветом, близким к белому, если их на изображении больше 10%
    if per_white > 10:
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                B = img[i][j][0]
                G = img[i][j][1]
                R = img[i][j][2]
                if (B > 110 and G > 110 and R > 110):
                    img[i][j] = [200,200,200]
        cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_2.jpg"), img)
        show(img,'step 2')

    #Guassian blur
    blur1 = cv2.GaussianBlur(img,(3,3),1)
    cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_3.jpg"), blur1)
    show(blur1,'step 3')
    
    #mean-shift algo
    newimg = np.zeros((img.shape[0], img.shape[1],3),np.uint8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 10 ,1.0)

    img = cv2.pyrMeanShiftFiltering(blur1, 20, 30, newimg, 0, criteria)
    
    cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_4.jpg"), img)
    show(img,'step 4 means shift image')

    #Guassian blur
    blur = cv2.GaussianBlur(img,(11,11),1)

    #Canny-edge detection
    canny = cv2.Canny(blur, 160, 290)
    
    cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_5.jpg"), canny)
    show(canny,'step 5 canny')

    canny = cv2.cvtColor(canny,cv2.COLOR_GRAY2BGR)
    
    #contour to find leafs
    bordered = cv2.cvtColor(canny,cv2.COLOR_BGR2GRAY)
    contours,hierarchy = cv2.findContours(bordered, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    maxC = 0
    for x in range(len(contours)):													#if take max or one less than max then will not work in
        if len(contours[x]) > maxC:													# pictures with zoomed leaf images
            maxC = len(contours[x])
            maxid = x

    perimeter = cv2.arcLength(contours[maxid],True)
    #print perimeter
    Tarea = cv2.contourArea(contours[maxid])
    cv2.drawContours(neworiginal,contours[maxid],-1,(0,0,255))
    
    cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_6.jpg"), neworiginal)
    show(canny,'step 6 Contour')

    #Creating rectangular roi around contour
    height, width, _ = canny.shape
    min_x, min_y = width, height
    max_x = max_y = 0
    frame = canny.copy()

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contours[maxid])
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)
        if w > 80 and h > 80:
            #cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)   #we do not draw the rectangle as it interferes with contour later on
            roi = img[y:y+h , x:x+w]
            originalroi = original[y:y+h , x:x+w]
            
    if (max_x - min_x > 0 and max_y - min_y > 0):
        roi = img[min_y:max_y , min_x:max_x]	
        originalroi = original[min_y:max_y , min_x:max_x]
        #cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)   #we do not draw the rectangle as it interferes with contour


    # cv2.imshow('ROI', frame)
    cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_7.jpg"), frame)
    show(canny,'step 7 ROI')
    
    # cv2.imshow('rectangle ROI', roi)
    cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_8.jpg"), roi)
    show(canny,'step 8 rectangle ROI')
    
    img = roi

    #Changing colour-space
    #imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    imghls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    
    # cv2.imshow('HLS', imghls)
    cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_9.jpg"), imghls)
    show(canny,'step 9 HLS')
   
'''    
    imghls[np.where((imghls==[30,200,2]).all(axis=2))] = [0,200,0]
    cv2.imshow('new HLS', imghls)

    #Only hue channel
    huehls = imghls[:,:,0]
    cv2.imshow('img_hue hls',huehls)
    #ret, huehls = cv2.threshold(huehls,2,255,cv2.THRESH_BINARY)

    huehls[np.where(huehls==[0])] = [35]
    cv2.imshow('img_hue with my mask',huehls)


    #Thresholding on hue image
    ret, thresh = cv2.threshold(huehls,28,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh', thresh)


    #Masking thresholded image from original image
    mask = cv2.bitwise_and(originalroi,originalroi,mask = thresh)
    cv2.imshow('masked out img',mask)


    #Finding contours for all infected regions
    contours,heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    Infarea = 0
'''

if __name__ == "__main__":
    main()
