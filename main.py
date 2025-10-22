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
    
    if os.path.isdir({cfg.results_folder}):
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
                img[i][j] = [200,200,200]
        cv2.imwrite(os.path.join(f"{cfg.results_folder}","output_step_2.jpg"), img)
        show(img,'step 2')
    

if __name__ == "__main__":
    main()
