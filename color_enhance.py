#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.exposure import histogram, cumulative_distribution
from scipy.stats import cauchy, logistic
#%%
M_f = np.array([[95.57, 64.67, 33.01],
            [49.49, 137.29, 14.76],
            [0.44, 27.21, 169.83]])
M_l = np.array([[4.61, 3.35, 1.78],
            [2.48, 7.16, 0.79],
            [0.28, 1.93, 8.93]])
M_cat02 = np.array([
    [0.7328, 0.4296, -0.1624],
    [-0.7036, 1.6975, 0.0061],
    [0.003, 0.0136, 0.9834]
])
M_h = np.array([
    [0.38971, 0.68898, -0.07868],
    [-0.22981, 1.18340, 0.04641],
    [0., 0., 1.]
])

def individual_channel(image, dist, channel):
    im_channel = img_as_ubyte(image[:,:,channel])
    freq, bins = cumulative_distribution(im_channel)
    new_vals = np.interp(freq, dist.cdf(np.arange(0,256)), 
                               np.arange(0,256))
    return new_vals[im_channel].astype(np.uint8)

def distribution(image, function, mean, std, output="value"):
    dist = function(mean, std)
    image_intensity = img_as_ubyte(rgb2gray(image))
    red = individual_channel(image, dist, 0)
    green = individual_channel(image, dist, 1)
    blue = individual_channel(image, dist, 2)

    if output=="value":
        return np.dstack((red, green, blue))

    elif output=="plot":
        fig, ax = plt.subplots(1,3, figsize=(8,5))
        freq, bins = cumulative_distribution(image_intensity)
        ax[0].step(bins, freq, c='b', label='Actual CDF')
        ax[0].plot(dist.cdf(np.arange(0,256)), 
                   c='r', label='Target CDF')
        ax[0].legend()
        ax[0].set_title('Actual vs. Target Cumulative Distribution')

        ax[1].imshow(image)
        ax[1].set_title('original Image')
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[2].imshow(np.dstack((red, green, blue)))
        ax[2].set_title('Transformed Image')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        plt.show()
    else:
        raise NameError("Output Type Neither 'value' or 'plot'")

def Gamma_correction(input, light_type, mapping=False):
    # if (input > 1.).any(): 
        # raise ValueError("Input Does Not Normalize")
    if light_type == "full-backlight":
        gamma = np.array([2.4767, 2.4286, 2.3792])
    elif light_type == "low-backlight":
        gamma = np.array([2.2212, 2.1044, 2.1835])
    else:
        raise TypeError("Image Type is Neither 'full-backlight' or 'low-backlight'")

    return np.power(input, 1/gamma) if mapping else np.power(input, gamma)

def xyz2rgb(xyz, light_type="low-backlight"):
    if light_type == "full-backlight":
        M_inv = np.linalg.inv(M_f)
    elif light_type == "low-backlight":
        M_inv = np.linalg.inv(M_l)
    else:
        raise NameError("Type Name Neither 'full-backlight' or 'low-backloght'")

    # M_inv = np.array([[3.2406, -1.5372, -0.4986],
                    # [-0.9689, 1.8758,  0.0415],
                    # [0.0557, -0.2040,  1.0570]])
    # xyz = xyz/100.0
    RGB = xyz.dot(M_inv.T)
    # RGB = np.where(RGB <= 0, 0.00000001, RGB)
    # RGB = np.where(RGB > 0.0031308,
                #    1.055*(RGB**0.4166666)-0.055,
                #    12.92*RGB)

    # RGB = RGB / 255.
    RGB = np.where(RGB < 0, 0, RGB)
    RGB = Gamma_correction(RGB, light_type, mapping=True)
    print(RGB)
    RGB = np.around(RGB*255)
    RGB = np.where(RGB <= 0, 0, RGB)
    RGB = np.where(RGB > 255, 255, RGB)
    RGB = RGB.astype('uint8')

    return RGB

def rgb2xyz(rgb, light_type="full-backlight"):
    if light_type == "full-backlight":
        M = M_f
    elif light_type == "low-backlight":
        M = M_l
    else:
        raise NameError("Type Name Neither 'full-backlight' or 'low-backloght'")
    
    rgb = Gamma_correction(rgb, light_type)
    # rgb = np.where(rgb > 0.04045, np.power(((rgb+0.055)/1.055), 2.4),
                    #  rgb/12.92)
    xyz = rgb.dot(M.T)
    return xyz

whitepoint = {'white': [193.25, 201.54, 197.48],
              'c': [109.85, 100.0, 35.58]}
env = {'dim': [0.9, 0.59, 0.9],
       'average': [1.0, 0.69, 1.0],
       'dark': [0.8, 0.525, 0.8]}
lightindensity = {'default': 60.0, 'high': 318.31, 'low': 31.83}
bgindensity = {'default': 25.0, 'high': 20.0, 'low': 10.0}

currentwhite = whitepoint['white']
currentenv = env['average']
currentlight = lightindensity['default']
currentbg = bgindensity['default']

def setconfig(wp='white', e='average', li='default', bgi='default'):
    currentwhite = whitepoint[wp]
    currentenv = env[e]
    currentlight = lightindensity[li]
    currentbg = bgindensity[bgi]

def xyz2cam02(xyz):
    # Xw, Yw, Zw = currentwhite
    Xw, Yw, Zw = rgb2xyz(np.array([1., 1., 1.]), "full-backlight")
    Nc, c, F = currentenv
    LA = currentlight
    Yb = currentbg
    Lw, Mw, Sw = M_cat02.dot(np.array([Xw, Yw, Zw]))
    D = F*(1 - (1/3.6)*np.exp(-(LA+42)/92))
    if D > 1: D = 1
    elif D < 0: D = 0

    Dl, Dm, Ds = [100*D/Lw+1-D, 100*D/Mw+1-D, 100*D/Sw+1-D]
    Lw_c, Mw_c, Sw_c = [Dl*Lw, Dm*Mw, Ds*Sw]
    k = 1 / (5*LA + 1)
    F_l = 0.2 * np.power(k, 4) * (5*LA) + 0.1 * np.power(1 - k ** 4, 2) * np.power(5*LA, 1/3)
    n = Yb/Yw
    if n > 1: n = 1
    elif n < 0: n = 1e-6

    Nbb = Ncb= 0.725 * np.power(1/n, 0.2)
    z = 1.48 + n ** 0.5

    Lw_, Mw_, Sw_ = M_h.dot(np.linalg.inv(M_cat02).dot([Lw_c, Mw_c, Sw_c]))

    Lwa_ = (400 * ((F_l*Lw_/100)**0.42))/(27.13+((F_l*Lw_/100)**0.42))+0.1
    Mwa_ = (400 * ((F_l*Mw_/100)**0.42))/(27.13+((F_l*Mw_/100)**0.42))+0.1
    Swa_ = (400 * ((F_l*Sw_/100)**0.42))/(27.13+((F_l*Sw_/100)**0.42))+0.1
    Aw = Nbb * (2*Lwa_+Mwa_+(Swa_/20) - 0.305)

    colordata = [
        [20.14, 0.8, 0],
        [90, 0.7, 100],
        [164.25, 1.0, 200],
        [237.53, 1.2, 300],
        [380.14, 0.8, 400]
    ]

    # step2
    LMS = xyz.dot(M_cat02.T)
    LMS_c = LMS * np.array([Dl, Dm, Ds])

    # step3
    LMS_ = LMS_c.dot(np.linalg.inv(M_cat02).T).dot(M_h.T)
    LMSa_ = (400*np.power(F_l*LMS_/100, 0.42)) / (27.13+np.power(F_l*LMS_/100, 0.42)) + 0.1

    # step4
    C1 = LMSa_[:, 0] - LMSa_[:, 1]
    C2 = LMSa_[:, 1] - LMSa_[:, 2]
    C3 = LMSa_[:, 2] - LMSa_[:, 1]

    A = (2*LMSa_[:, 0] + LMSa_[:, 1] + 1/20*LMSa_[:, 2] - 0.305) * Nbb
    a = C1 - 1/11*C2
    b = (C2-C3) / 9

    # step5
    h = np.arctan2(b, a)
    h = np.where(h < 0, (h+np.pi*2)*180/np.pi, h*180/np.pi)
    huue = np.where(h < colordata[0][0], h+360, h)
    etemp = (np.cos(huue*np.pi/180+2)+3.8) * 0.25
    coarray = np.array([20.14, 90, 164.25, 237.53, 380.14])
    position_ = coarray.searchsorted(huue)

    def TransferHue(h_, i):
        datai = colordata[i-1]
        datai1 = colordata[i]
        Hue = datai[2] + ((100*(h_-datai[0])/datai[1]) /
                          (((h_-datai[0])/datai[1])+(datai1[0]-h_)/datai1[1]))
        return Hue

    ufunc_TransferHue = np.frompyfunc(TransferHue, 2, 1)
    H = ufunc_TransferHue(huue, position_).astype('float')
    J = 100*((A/Aw)**(c*z))
    Q = (4/c) * ((J/100.0)**0.5) * (Aw + 4) * (F_l**0.25)
    # step 12
    t = ((50000/13.0)*Nc*Nbb*etemp*((a**2+b**2)**0.5)) / (LMSa_[:, 0]+LMSa_[:, 1]+(21/20.0)*LMSa_[:, 2])
    C = t**0.9*((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)
    M = C*(F_l**0.25)
    s = 100*((M/Q)**0.5)
    return np.array([h, H, J, Q, C, M, s]).T

def jch2xyz(jch):
    # Xw, Yw, Zw = currentwhite
    Xw, Yw, Zw = rgb2xyz(np.array([1., 1., 1.]), "low-backlight")
    Nc, c, F = currentenv
    LA = currentlight
    Yb = currentbg
    Lw, Mw, Sw = M_cat02.dot(np.array([Xw, Yw, Zw]))
    D = F*(1 - (1/3.6)*np.exp(-(LA+42)/92))
    if D > 1: D = 1
    elif D < 0: D = 0

    Dl, Dm, Ds = [100*D/Lw+1-D, 100*D/Mw+1-D, 100*D/Sw+1-D]
    Lw_c, Mw_c, Sw_c = [Dl*Lw, Dm*Mw, Ds*Sw]
    k = 1 / (5*LA + 1)
    F_l = 0.2 * np.power(k, 4) * (5*LA) + 0.1 * np.power(1 - k ** 4, 2) * np.power(5*LA, 1/3)
    n = Yb/Yw
    if n > 1: n = 1
    elif n < 0: n = 1e-6

    Nbb = Ncb= 0.725 * np.power(1/n, 0.2)
    z = 1.48 + n ** 0.5

    Lw_, Mw_, Sw_ = M_h.dot(np.linalg.inv(M_cat02).dot([Lw_c, Mw_c, Sw_c]))

    Lwa_ = (400 * ((F_l*Lw_/100)**0.42))/(27.13+((F_l*Lw_/100)**0.42))+0.1
    Mwa_ = (400 * ((F_l*Mw_/100)**0.42))/(27.13+((F_l*Mw_/100)**0.42))+0.1
    Swa_ = (400 * ((F_l*Sw_/100)**0.42))/(27.13+((F_l*Sw_/100)**0.42))+0.1
    Aw = Nbb * (2*Lwa_+Mwa_+(Swa_/20) - 0.305)

    colordata = [
        [20.14, 0.8, 0],
        [90, 0.7, 100],
        [164.25, 1.0, 200],
        [237.53, 1.2, 300],
        [380.14, 0.8, 400]
    ]

    JCH = jch*np.array([1.0, 1.0, 10/9.0])
    J = JCH[:, 0]
    C = JCH[:, 1]
    H = JCH[:, 2]
    coarray = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    position_ = coarray.searchsorted(H)

    def TransferHue(H_, i):
        C1 = colordata[i-1]
        C2 = colordata[i]
        h = ((H_-C1[2])*(C2[1]*C1[0]-C1[1]*C2[0])-100*C1[0]*C2[1]) /\
            ((H_-C1[2])*(C2[1]-C1[1]) - 100*colordata[i][1])
        if h > 360:
            h -= 360
        return h
    ufunc_TransferHue = np.frompyfunc(TransferHue, 2, 1)
    h_ = ufunc_TransferHue(JCH[:, 2], position_).astype('float')
    J = np.where(J <= 0, 0.00001, J)
    C = np.where(C <= 0, 0.00001, C)
    t = (C/(((J/100.0)**0.5)*((1.64-(0.29**n))**0.73)))**(1/0.9)
    t = np.where(t - 0 < 0.00001, 0.00001, t)
    etemp = (np.cos(h_*np.pi/180+2)+3.8) * 0.25
    e = (50000/13.0) * Nc * Nbb * etemp
    A = Aw*((J/100)**(1/(c*z)))

    pp2 = A/Nbb + 0.305
    p3 = 21/20.0
    hue = h_*np.pi/180
    pp1 = e/t

    def evalAB(h, p1, p2):
        if abs(np.sin(h)) >= abs(np.cos(h)):
            p4 = p1/np.sin(h)
            b = (p2*(2+p3)*(460.0/1403)) /\
                (p4+(2+p3)*(220.0/1403)*(np.cos(h)/np.sin(h))-27.0/1403 +
                 p3*(6300.0/1403))
            a = b*(np.cos(h)/np.sin(h))
        else:  # abs(np.cos(h))>abs(np.sin(h)):
            p5 = p1/np.cos(h)
            a = (p2*(2+p3)*(460.0/1403)) /\
                (p5+(2+p3)*(220.0/1403) -
                 (27.0/1403 - p3*(6300.0/1403))*(np.sin(h)/np.cos(h)))
            b = a*(np.sin(h)/np.cos(h))
        return np.array([a, b])
    ufunc_evalAB = np.frompyfunc(evalAB, 3, 1)
    abinter = np.row_stack(ufunc_evalAB(hue, pp1, pp2))
    a = abinter[:, 0]
    b = abinter[:, 1]

    Ra_ = (460*pp2 + 451*a + 288*b)/1403.0
    Ga_ = (460*pp2 - 891*a - 261*b)/1403.0
    Ba_ = (460*pp2 - 220*a - 6300*b)/1403.0
    R_ = np.sign(Ra_-0.1)*(100.0/F_l) *\
        (((27.13*np.abs(Ra_-0.1))/(400-np.abs(Ra_-0.1)))**(1/0.42))
    G_ = np.sign(Ga_-0.1)*(100.0/F_l) *\
        (((27.13*np.abs(Ga_-0.1))/(400-np.abs(Ga_-0.1)))**(1/0.42))
    B_ = np.sign(Ba_-0.1)*(100.0/F_l) *\
        (((27.13*np.abs(Ba_-0.1))/(400-np.abs(Ba_-0.1)))**(1/0.42))

    RcGcBc = (np.array([R_, G_, B_]).T).dot(np.linalg.inv(M_h).T).dot(M_cat02.T)
    RGB = RcGcBc/np.array([Dl, Dm, Ds])
    XYZ = RGB.dot(np.linalg.inv(M_cat02).T)
    return XYZ

def rgb2jch(color, light_type="full-backlight"):
    XYZ = rgb2xyz(color, light_type)
    value = xyz2cam02(XYZ)
    return value[:, [2, 4, 1]]*np.array([1.0, 1.0, 0.9])

def jch2rgb(jch, light_type="low-backlight"):
    xyz = jch2xyz(jch)
    return xyz2rgb(xyz, light_type)

def clipped(jch, rgb_c, rgb_i):
    J = np.array([jch[:, 0]]).T
    C = np.array([jch[:, 1]]).T
    JC = J*C
    JC = (JC - JC.min()) / (JC.max()-JC.min())
    RGB = (1 - JC)*(rgb_c/255.) + (JC)*(rgb_i)

    RGB = np.where(RGB <= 0, 0, RGB)
    RGB = np.where(RGB > 1, 1, RGB)
    RGB = np.around(RGB*255)
    RGB = RGB.astype('uint8')
    return RGB

# %%
# img = Image.open("image_ref/04_original.png")
imageObj = plt.imread('image_ref/18_original.png')
# distribution(imageObj, logistic, 120, 40, "plot")
hist_img = distribution(imageObj, logistic, 120, 40, "value")
plt.imshow(hist_img)
#%%
rgb = np.array(hist_img)
rgb = rgb / 255.
shape = rgb.shape
jch = rgb2jch(rgb.reshape(-1, 3), "low-backlight")
enhanced_rgb = jch2rgb(jch, "full-backlight").reshape(shape)
enhanced_img = Image.fromarray(enhanced_rgb)
# enhance_im.save("18_dim.png")
enhanced_img.show()
#%%
clip = clipped(jch, enhanced_rgb.reshape(-1, 3), rgb.reshape(-1, 3)).reshape(shape)
clip_img = Image.fromarray(clip)
clip_img.show()
#%%
fig, ax = plt.subplots(1,2, figsize=(8,5))
ax[0].imshow(hist_img)
ax[0].set_title('original Image')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].imshow(enhanced_img)
ax[1].set_title('Enhanced Image')
ax[1].set_xticks([])
ax[1].set_yticks([])
# ax[2].imshow(clip_img)
# ax[2].set_title('Clipped Image')
# ax[2].set_xticks([])
# ax[2].set_yticks([])