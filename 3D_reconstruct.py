import os
import cv2
import numpy as np
from numpy import sqrt, arctan2, pi, sin, cos, abs
import matplotlib.pyplot as plt
from triangulation import triangulation

def read_calib_param(param_txt):
    with open(param_txt, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    intrinsic_1 = np.array([[float(lines[0]), float(lines[1]), float(lines[2])],
                            [float(lines[3]), float(lines[4]), float(lines[5])],
                            [float(lines[6]), float(lines[7]), float(lines[8])]])
    kc1 = np.array([ float(lines[9]), 
                     float(lines[10]), 
                     float(lines[11]), 
                     float(lines[12]), 
                     float(lines[13])])
    intrinsic_2 = np.array([[float(lines[14]), float(lines[15]), float(lines[16])],
                            [float(lines[17]), float(lines[18]), float(lines[19])],
                            [float(lines[20]), float(lines[21]), float(lines[22])]])
    kc2 = np.array([ float(lines[23]), 
                     float(lines[24]), 
                     float(lines[25]), 
                     float(lines[26]), 
                     float(lines[27])])
    R = np.array([ [float(lines[28]), float(lines[29]), float(lines[30])],
                   [float(lines[31]), float(lines[32]), float(lines[33])],
                   [float(lines[34]), float(lines[35]), float(lines[36])]])
    T = np.array([ [float(lines[37])],
                   [float(lines[38])],
                   [float(lines[39])]])

    return intrinsic_1, kc1, intrinsic_2, kc2, R, T

def phase_shift_4(images):
    a = images[3].astype(np.int16) - images[1].astype(np.int16)
    b = images[0].astype(np.int16) - images[2].astype(np.int16)
    
    thresh = 10

    r = sqrt(a * a + b * b) + 0.5
    phase = pi + arctan2(a, b)
    mask = r>=thresh
    return phase, mask

def phase_shift_6(images):
    b = images[3] * sin(0 * 2* pi / 6.0) \
      + images[4] * sin(1 * 2* pi / 6.0) \
      + images[5] * sin(2 * 2* pi / 6.0) \
      + images[0] * sin(3 * 2* pi / 6.0) \
      + images[1] * sin(4 * 2* pi / 6.0) \
      + images[2] * sin(5 * 2* pi / 6.0)


    a = images[3] * cos(0 * 2* pi / 6.0) \
      + images[4] * cos(1 * 2* pi / 6.0) \
      + images[5] * cos(2 * 2* pi / 6.0) \
      + images[0] * cos(3 * 2* pi / 6.0) \
      + images[1] * cos(4 * 2* pi / 6.0) \
      + images[2] * cos(5 * 2* pi / 6.0)
      
    #saturate_mask = (images[0]==255 | images[1]==255 | images[2]==255 | images[3]==255 | images[4]==255 | images[5]==255)
    thresh = 10

    r = sqrt(a * a + b * b) + 0.5
    phase = pi + arctan2(a, b)
    mask = r>=thresh
    return phase, mask

def phase_unwrap(last_phase, next_phase, scale_factor):
    predicted_phase = last_phase * scale_factor
    nth_period = (predicted_phase - next_phase) / (2 * pi)
    k = nth_period.round()
    corrected_phase = 2 * pi * k + next_phase
    corrected_phase[next_phase==-1] = -1
    error = abs(corrected_phase - predicted_phase)
    return corrected_phase

    
def read_images(image_dir):
    images = []
    for i in range(36):
        image = cv2.imread(os.path.join(image_dir, 'phase%02d.bmp'%i), 0)
        images.append(image)
    return images

def show_pseudo_color_image(image, save_file = 'result.png'):
    max_value = np.max(image)
    min_value = np.min(image)  
    image_color = (image - min_value) / (max_value - min_value) * 255
    image_color = image_color.astype(np.uint8)
    image_color = cv2.applyColorMap(image_color, cv2.COLORMAP_JET)
    image_color[image<0] = [0,0,0]
    cv2.imwrite(save_file, image_color)
    cv2.namedWindow('image', 0)
    cv2.imshow('image', image_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_images(images):
    for i in range(len(images)):
        cv2.imshow('image%d'%i, images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_phase(images):
    phase1, mask1 = phase_shift_4(images[0:4])
    show_pseudo_color_image(phase1, 'result1.png')
    phase2, mask2 = phase_shift_4(images[4:8])
    show_pseudo_color_image(phase2, 'result2.png')
    phase_wrapped = phase_unwrap(last_phase=phase1, next_phase=phase2, scale_factor=8)
    
    phase3, mask3 = phase_shift_4(images[8:12])
    show_pseudo_color_image(phase3, 'result3.png')
    phase_wrapped = phase_unwrap(last_phase=phase_wrapped, next_phase=phase3, scale_factor=4)
    
    phase4, mask4 = phase_shift_6(images[12:18])
    show_pseudo_color_image(phase4, 'result4.png')
    phase_wrapped = phase_unwrap(last_phase=phase_wrapped, next_phase=phase4, scale_factor=4)
    
    mask = mask1 & mask2 & mask3 & mask4
    
    return phase_wrapped, mask

def get_DMD_coordinate(images):
    phase, mask = get_phase(images)
    DMD_coordinate = phase / (2 * pi) * 10
    return DMD_coordinate, mask


def reconstruct(params_path, raw_image_dir, output_path):
    left_intrinsic, kc_left, right_intrinsic, kc_right, R, T = read_calib_param(params_path)

    images = read_images(raw_image_dir)
    DMD_x, mask_x = get_DMD_coordinate(images[0:18])
    DMD_y, mask_y = get_DMD_coordinate(images[18:36])
    mask = mask_x & mask_y

    xR = np.array(list(zip(DMD_x[mask], DMD_y[mask])), dtype=np.float32)
    X,Y = np.meshgrid(np.arange(0,1920), np.arange(0,1200))
    xL = np.array(list(zip(X[mask], Y[mask])), dtype=np.float32)

    XL, XR, ERR = triangulation( xL, xR, 
                                        R,T,
                                        left_intrinsic,  kc_left,  0,
                                        right_intrinsic, kc_right, 0)
                                        
    np.savetxt(output_path, XL, fmt='%.4f', delimiter=',', newline='\n')

if __name__ == '__main__':
    reconstruct('raw/param.txt', 'raw', 'point_cloud.xyz')