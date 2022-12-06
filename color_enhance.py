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

def CIECAM(X, Y, Z, surround="avg"):
    if surround == "avg":
        c = 0.69
        N = 1.0
        F = 1.0
    elif surround == "dim":
        c = 0.59
        N = 0.9
        F = 0.9
    elif surround == "dark":
        c = 0.525
        N = 0.8
        F = 0.8
    else:
        raise NameError("Surround Neither 'avg' or 'dim' or 'dark'")

    white = np.ones(X.shape)
    R_w = Gamma_correction(white, 2.4767)
    G_w = Gamma_correction(white, 2.4286)
    B_w = Gamma_correction(white, 2.3792)
    X_w, Y_w, Z_w = DCM(R_w, G_w, B_w, type="full-backlight")
    L_a = 60
    Y_b = 25
    M_cat = np.array([
        [0.7328, 0.4296, -0.1624],
        [-0.7036, 1.6975, 0.0061],
        [0.003, 0.0136, 0.9834]
    ])

    # step2
    L = X*M_cat[0, 0] + Y*M_cat[0, 1] + Z*M_cat[0, 2]
    M = X*M_cat[1, 0] + Y*M_cat[1, 1] + Z*M_cat[1, 2]
    S = X*M_cat[2, 0] + Y*M_cat[2, 1] + Z*M_cat[2, 2]
    L_w = X_w*M_cat[0, 0] + Y_w*M_cat[0, 1] + Z_w*M_cat[0, 2]
    M_w = X_w*M_cat[1, 0] + Y_w*M_cat[1, 1] + Z_w*M_cat[1, 2]
    S_w = X_w*M_cat[2, 0] + Y_w*M_cat[2, 1] + Z_w*M_cat[2, 2]

    D = F*(1 - (1/3.6)*np.exp(-(L_a+42)/92))
    L_c = ((100/L_w)*D + 1 - D) * L
    M_c = ((100/M_w)*D + 1 - D) * M
    S_c = ((100/S_w)*D + 1 - D) * S

    # step3
    k = 1 / (5*L_a + 1)
    F_l = 0.2 * np.power(k, 4) * (5*L_a) + 0.1 * np.power(1 - k ** 4, 2) * np.power(5*L_a, 1/3)
    M_h = np.array([
        [0.38971, 0.68898, -0.07868],
        [-0.22981, 1.18340, 0.04641],
        [0., 0., 1.]
    ])
    X_c = L_c*np.linalg.inv(M_cat)[0, 0] + M_c*np.linalg.inv(M_cat)[0, 1] + S_c*np.linalg.inv(M_cat)[0, 2]
    Y_c = L_c*np.linalg.inv(M_cat)[1, 0] + M_c*np.linalg.inv(M_cat)[1, 1] + S_c*np.linalg.inv(M_cat)[1, 2]
    Z_c = L_c*np.linalg.inv(M_cat)[2, 0] + M_c*np.linalg.inv(M_cat)[2, 1] + S_c*np.linalg.inv(M_cat)[2, 2]
    L_ = X_c*M_h[0, 0] + Y_c*M_h[0, 1] + Z_c*M_h[0, 2]
    M_ = X_c*M_h[1, 0] + Y_c*M_h[1, 1] + Z_c*M_h[1, 2]
    S_ = X_c*M_h[2, 0] + Y_c*M_h[2, 1] + Z_c*M_h[2, 2]

    La_ = (400*np.power(F_l*L_/100, 0.42)) / (27.13+np.power(F_l*L_/100, 0.42)) + 0.1
    Ma_ = (400*np.power(F_l*M_/100, 0.42)) / (27.13+np.power(F_l*M_/100, 0.42)) + 0.1
    Sa_ = (400*np.power(F_l*S_/100, 0.42)) / (27.13+np.power(F_l*S_/100, 0.42)) + 0.1

    # step4
    n = Y_b/Y_c
    N_bb = 0.725 * np.power(1/n, 0.2)
    C1 = La_ - Ma_
    C2 = Ma_ - Sa_
    C3 = Sa_ - La_

    A = (2*La_ + Ma_ + 1/20*Sa_ - 0.305) * N_bb
    a = C1 - 1/11*C2
    b = (C2-C3) / 9

    # step5
    z = 1.48 + n ** 0.5

    lightness = 100 * np.power(A/A_w, c*z)
    chroma = ...
    hue = ...
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
#%%