import hydra
from omegaconf import DictConfig
import cv2
import numpy as np           
import sys, os

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
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
    cv2.imshow('original',img)


if __name__ == "__main__":
    main()
