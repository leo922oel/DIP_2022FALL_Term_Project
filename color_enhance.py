#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
def Gamma_correction(input, gamma):
    if (input > 1.).any(): 
        raise ValueError("Input Does Not Normalize")
    return np.power(input, 1/gamma)

def DCM(R, G, B, type="full-backlight"):
    if type == "full-backlight":
        M = np.array([
            [95.57, 64.67, 33.01],
            [49.49, 137.29, 14.76],
            [0.44, 27.21, 169.83]
        ])
    elif type == "low-backlight":
        M = np.array([
            [4.61, 3.35, 1.78],
            [2.48, 7.16, 0.79],
            [0.28, 1.93, 8.93]
        ])
    else:
        raise NameError("Type Name Neither 'full-backlight' or 'low-backloght'")
    
    X = R*M[0, 0] + G*M[0, 1] + B*M[0, 2]
    Y = R*M[1, 0] + G*M[1, 1] + B*M[1, 2]
    Z = R*M[2, 0] + G*M[2, 1] + B*M[2, 2]
    return X, Y, Z

def CIECAM():
    pass
#%%
ori_img = cv2.imread("image_ref/04_original.png")
norm_img = ori_img / 255.
R = norm_img[:, :, 0]
G = norm_img[:, :, 1]
B = norm_img[:, :, 2]

# full-backlight
Gamma_r = Gamma_correction(R, 2.4767)
Gamma_g = Gamma_correction(G, 2.4286)
Gamma_b = Gamma_correction(B, 2.3792)

DCM(Gamma_r, Gamma_g, Gamma_b, "full-backlight")

#%%
# cv2.imshow('My Image', ori_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
