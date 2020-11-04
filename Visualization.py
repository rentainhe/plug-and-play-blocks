import PIL
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage
import skimage.transform

'''
    Visualization操作: 将图片和attention同时打印出来
'''

def visulize_attention(img,alphas):
        """
        alphas:     attn weights, (49,)
        img_path:   image file path to load
        save_path:  image file path to save
        scale:      reshape alphas to (scale, scale)
        return:
        """
        # load the image
        # img = skimage.img_as_float(img).astype(np.float32)
#         img_h, img_w = img.shape[0], img.shape[1]
        img_h, img_w = img.size[0], img.size[1]
        plt.subplots(nrows=1, ncols=1, figsize=(0.01 * img_h, 0.01 * img_w))

        # cv2.resize(img,(224,224))
        # alphas = np.array(alphas).swapaxes(0, 1)  # n,49,1
        plt.imshow(img)
        plt.axis('off')
#         scale = 50
        # alpha_img = skimage.transform.pyramid_expand(alphas[0,:].reshape(7,7),upscale=16,sigma=20)
        # alpha_img = skimage.transform.resize(alphas[0, :].reshape(scale, scale), [img.shape[0], img.shape[1]])
        print(img_w)
        print(img_h)
        alpha_img = skimage.transform.resize(alphas, ([img_w, img_h]))
        plt.imshow(alpha_img, alpha=0.5, interpolation='nearest')
        # plt.set_cmap(cm.Greys_r)
#         plt.set_cmap(cm.jet)
#         # plt.axis('off')

#         # adjust the figure
#         plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)

        # save
        # img_name = img_path.split('/')[-1]
#         plt.savefig(save_path + 'atted.jpg', format='jpg', dpi=100)