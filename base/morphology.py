# coding=utf-8

# Third Party Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt


x,y,w,h = 0,0,0,0
pix2um = 3.45 / 40 * 2
wavelength = 0.53


def im2contour(i, org_img, background):
    norm_img = cv2.normalize(org_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    img = norm_img - background # 90 for RBC/WBC, 150 for PLT
    print(np.min(img), np.mean(img), np.max(img))
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 45, 100) # 45, 70 for RBC/WBC, 63, 100 for PLT
    contours, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(org_img, contours, -1, (255),-1)
    cv2.fillPoly(thresh,pts=contours,color=(255,255,255))
    return contours, img


def morph(contour):       
    area = cv2.contourArea(contour) * pix2um**2
    area = cv2.contourArea(contour) * pix2um**2
    xa, ya, wa, ha = cv2.boundingRect(contour)
    r = np.sqrt(area / np.pi)
    try:
        aRatio = float(wa)/ha
    except ZeroDivisionError:
        aRatio = np.inf
    pVolume = (4/3) * np.pi * (r ** 3)
    return 2*r, aRatio, area, pVolume


def perimeter(contour):
    return cv2.arcLength(contour, True) * pix2um


def circularity(peri, area): 
    try:
        c_index = 4*np.pi*(area/(peri*peri))  
    except ZeroDivisionError:
        c_index = np.inf
    return c_index  


def sphericity(area):
    try:
        s_index = ((np.pi**(1/3))*(6*(area/(pix2um**2)))**(2/3))/(area/(pix2um**2))
    except ZeroDivisionError:
        s_index = np.inf
    return s_index


def optics(org_img, contours, i):
    cimg = np.zeros_like(org_img)
    cv2.drawContours(cimg, contours, i, color=(255,0,0), thickness=-1)
    plt.imshow(cimg)
    pts = np.where(cimg == 255)

    pShift = np.mean(org_img[pts[0], pts[1]]) # the phase angle/shift
    oHeight = pShift * wavelength / (2 * np.pi) # the optical height: multiply with lambda/2pi
    pHeight = oHeight / (1.42-1.33) # the physical height: multiply with 1 / (nc - nm)
    oVolume = float(np.sum(cimg)) * wavelength * 0.01 / (2 * np.pi) * pix2um ** 2

    x, y, w, h = cv2.boundingRect(contours[i])
    bbox = [x-1, y-1, w+1, h+1]

    return pShift, oHeight, pHeight, oVolume, cimg, bbox


def img_thresholding(img, contours,index, img_index, output_path, experiment_name, save_csv):
    global pix2um,wavelength,x,y,w,h

    eq_diameter, aRatio, area, pVolume = morph(contours[index])
    
    peri = perimeter(contours[index])
    c_index = circularity(peri, area) 
    s_index = sphericity(area)
    if c_index > 0.5:
        if area > 1 and area < 30: # if area > 7 and area < 30
            pShift, oHeight, pHeight, oVolume, cimg, bbox = optics(img, contours, index)
            
            feature_list = [eq_diameter, aRatio, area, pVolume, peri, c_index, s_index, pShift, oHeight, pHeight, oVolume]

            # cv2.imwrite(f'{output_path}/frame_{img_index}_cell_{index}.png', norm_img)
            if save_csv:
                write_to_csv(img_index, index, experiment_name, feature_list, output_path)
            else:
                pass


def write_to_csv(img_index, index, experiment_name, feature_list, output_path):
    eq_diameter, aRatio, area, pVolume, peri, c_index, s_index, pShift, oHeight, pHeight, oVolume = feature_list
    with open(f'{output_path}/{experiment_name}.csv', 'a') as f:
        list = [img_index, index, eq_diameter, aRatio, area, pVolume, peri, c_index, s_index, pShift, oHeight, pHeight, oVolume]
        for i in list:
            f.write(str(i))
            f.write(',')
        f.write('\n')


def qc_img_thresholding(img, contours, index):
    global pix2um,wavelength,x,y,w,h

    eq_diameter, aRatio, area, pVolume = morph(contours[index])
    feature_list = []
    
    peri = perimeter(contours[index])
    c_index = circularity(peri, area) 
    s_index = sphericity(area)
    if c_index > 0.5:
        if area > 1 and area < 30: # if area > 7 and area < 30
            pShift, oHeight, pHeight, oVolume, cimg, bbox = optics(img, contours, index)
            
            feature_list = [eq_diameter, aRatio, area, pVolume, peri, c_index, s_index, pShift, oHeight, pHeight, oVolume]
            
    return feature_list
