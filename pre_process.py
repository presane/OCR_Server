#c10_cuda.dll error
# remove c:\Windows\System32\libiomp5md.dll
# install คำสั่ง ติดตั้ง  pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html    แล้วติดตั้ง pytesseract
import cv2
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter

def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def skew_correct(img):
    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = find_score(img, angle)
        scores.append(score)

    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, \
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

############################################################################################################
# Filter Image  2020-08-22 Preecha J. #####################################################################
def preprocess_img(img, name):
    
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 40) 
    #kernel = np.ones((2,2),np.uint8)
    #img = cv2.dilate(img,kernel,iterations = 1)
    # Zoom in twice, easier to identify
    resize_img = cv2.resize(img, (int(5.0 * img.shape[1]), int(5.0 * img.shape[0])), interpolation=cv2.INTER_CUBIC)
    #plt.imshow(resize_img,'gray')
    resize_img = cv2.convertScaleAbs(resize_img, alpha=0.35, beta=40) 
    resize_img = cv2.normalize(resize_img, dst=None, alpha=300, beta=10, norm_type=cv2.NORM_MINMAX)
    
    # Median filtering
    img_blurred = cv2.medianBlur(resize_img, 7)
    img_blurred = cv2.medianBlur(img_blurred, 3)
    
    #kernel = np.ones((5,5),np.uint8)
    #img_blurred = cv2.dilate(img_blurred,kernel,iterations = 1)
    
    img_blurred = cv2.cvtColor(img_blurred,cv2.COLOR_BGR2GRAY)
    ret,img_blurred = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5),np.uint8)
    img_blurred = cv2.dilate(img_blurred,kernel,iterations = 1)
    
    return img_blurred
############################################################################################################
