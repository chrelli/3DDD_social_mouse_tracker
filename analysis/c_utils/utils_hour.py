# FROM TIRAMISU
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# import the tiramisu models
#from pytorch_tiramisu.models import tiramisu

# from pose.models import hourglass
#import deepfly.pose2d.models as flymodels

# from datasets import camvid
#from pytorch_tiramisu.datasets import joint_transforms
#import pytorch_tiramisu.utils.imgs
#import pytorch_tiramisu.utils.training as train_utils

import sys, os, pickle
import h5py
import cv2
from colour import Color

#%%

# for making the target maps!
def gaussian(img, pt, sigma):
    # Draw a 2D gaussian, unless the point is in the upper corner

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] > img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0) :
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

#%%
def check_h5(h5_path):
    # plots a random file from the h5py
    with h5py.File(h5_path, mode='r') as h5file:
        print(h5file.keys())
        ji = np.random.choice(len(h5file['c_images']))
        c_image = h5file['c_images'][ji]
        points = h5file['annotations'][ji]

    plt.figure(figsize=(10,10))

    plt.imshow(c_image[..., [2,1,0]])
    plt.plot(points[:,0],points[:,1],'or')
    plt.title("raw width: {} height: {}".format(c_image.shape[0],c_image.shape[1]))
    plt.show()

    h5file.close()
    
#%%
def check_h5_ir(h5_path, ji = None):
    # plots a random file from the h5py
    with h5py.File(h5_path, mode='r') as h5file:
        print(h5file.keys())
        if ji is None:
            ji = np.random.choice(len(h5file['c_images']))
        c_image = h5file['c_images'][ji]
        points = h5file['annotations'][ji]

    plt.figure(figsize=(10,10))

    plt.imshow(c_image)
    plt.plot(points[:,0],points[:,1],'or')
    plt.title("raw width: {} height: {}".format(c_image.shape[0],c_image.shape[1]))
    plt.show()

    h5file.close()
    return c_image

#%%
def check_h5_ir_bw(h5_path, ji = None,savepath=None):
    # plots a random file from the h5py
    with h5py.File(h5_path, mode='r') as h5file:
        print(h5file.keys())
        if ji is None:
            ji = np.random.choice(len(h5file['c_images']))
        c_image = h5file['c_images'][ji]
        points = h5file['annotations'][ji]
        annotated = h5file['annotated'][ji]
        skel = h5file['skeleton'][:]
        
        
        print(h5file.keys())
        print(skel)
    # housekeeping for plotting
    body_colors =['dodgerblue','red','lime','orange']    
    label_names = ['impl','ear','ear','nose','tail','ear','ear','nose','tail']
    body_names = ['mouse0','mouse0','mouse0','mouse0','mouse0','mouse1','mouse1','mouse1','mouse1']
    label_index = [0,1,1,2,3,1,1,2,3]
    body_index = [0,0,0,0,0,1,1,1,1]

    
    
    plt.figure(figsize=(10,10))

    plt.imshow(c_image,cmap = 'gray')
    
    for jj in range(points.shape[0]):
        if points[jj,0] <10:
            continue
        cc =body_colors[label_index[jj]]
        plt.scatter(points[jj,0],points[jj,1],marker='o',s=200,edgecolor=cc,facecolor='none',linewidth=3)
    
    
#     plt.title("raw width: {} height: {}".format(c_image.shape[0],c_image.shape[1]))
    plt.axis('off')
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

    h5file.close()

    return c_image,points


#%% SOME PLOTTING
def plot_im_target(im,target,size = 5):
    im_np = im.numpy()
    target_np = target.numpy()[0,:,:]
    c = im_np[0,[2,1,0],:,:]
#     dac = im_np[0,3,:,:]
    c = np.moveaxis(c,[0],[2])
    #     dac = im_c[]

    point_map = np.max( target_np[:4,:,:] , axis = 0)
    posture_map = np.max( target_np[4:,:,:] , axis = 0)
    full_map = np.max( target_np[:,:,:] , axis = 0)
    # plt.imshow(posture_map)

    plt.figure(figsize=(1.3*size,size))
    plt.subplot(2,2,1)
    plt.imshow( c )
    plt.title("RGB")

    plt.subplot(2,2,2)
    plt.imshow( full_map )
    
#     plt.imshow( dac )
    plt.title("all")

    plt.subplot(2,2,3)
    plt.imshow( point_map )
    plt.title("Point targets")

    plt.subplot(2,2,4)
    plt.imshow( posture_map )
    plt.title("Affinity map")

    plt.show()

def plot_im_target_ir(im,target,size = 5):
    im_np = im.numpy()
    target_np = target.numpy()[0,:,:]
    c = im_np[0,0,:,:]
#     dac = im_np[0,3,:,:]
#     c = np.moveaxis(c,[0],[2])
    #     dac = im_c[]

    point_map = np.max( target_np[:4,:,:] , axis = 0)
    posture_map = np.max( target_np[4:,:,:] , axis = 0)
    full_map = np.max( target_np[:,:,:] , axis = 0)
    # plt.imshow(posture_map)

    plt.figure(figsize=(1.3*size,size))
    plt.subplot(2,2,1)
    plt.imshow( c )
    plt.title("RGB")

    plt.subplot(2,2,2)
    plt.imshow( full_map )
    
#     plt.imshow( dac )
    plt.title("all")

    plt.subplot(2,2,3)
    plt.imshow( point_map )
    plt.title("Point targets")

    plt.subplot(2,2,4)
    plt.imshow( posture_map )
    plt.title("Affinity map")

    plt.show()    
    
    
def random_from(MouseValidLoader):
    N = MouseValidLoader.__len__()
    k = np.random.randint(0,N)
    for i, data in enumerate(MouseValidLoader):
        if i == k:
            print(i)
            return data[0],data[1]

def specific_from(MouseValidLoader,k):
    N = MouseValidLoader.__len__()
    for i, data in enumerate(MouseValidLoader):
        if i == k:
            print(i)
            return data[0],data[1]        
        
def plot_im_target_pseudo(input_var,target_var,size = 10,save_fig = False):
    # def show_frame(input_var,target_var):
    # plt.imshow(input_var.data.cpu()[0,:,:,:].numpy())
    input_image = input_var.data.cpu()[0,:3,:,:].numpy()
    input_image = np.moveaxis(input_image,0,2)

    target_stack = target_var.data.cpu()[0,:,:,:].numpy()

    target_image = target_stack[:4,...]
    target_pose = target_stack[4:,...]

    # = np.moveaxis(target[0,:,:,:].numpy() ,0,2)
    # test.shape
    # score_map = output.data.cpu()


    tt = ["implant","ears","noses",'tails']


    # show the tracking belief map
    Fig1 = plt.figure(figsize=(1.5*size,size))


    plt.subplot(2,3,1)
    plt.imshow(input_image[:,:,[2,1,0]])
    plt.title("image space, h: {} w: {}".format(input_image.shape[0],input_image.shape[1]) )

    plt.subplot(2,3,2)
    # from matplotlib.pyplot import cm

    pseudo = np.zeros((target_image.shape[1],target_image.shape[2],3))

    body_colors =['dodgerblue','red','lime','orange']
    for i,col in enumerate(body_colors):
        bright = target_image[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        pseudo += color_im

    pseudo = np.clip(pseudo,0,1)

    # # Write some Text
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = .4
    # lineType = 0
    t_h,t_w = pseudo.shape[:2]
    pad = 10
    # for i,(type,col,x,y) in enumerate(zip(tt,body_colors,[pad,pad,t_h-pad,t_h-pad],[pad,2*pad,pad,2*pad])):
    # # do as in-place?
    #     rgb = Color(col).rgb
    #     fontColor = rgb
    #     bottomLeftCornerOfText = (10,i*10+20)
    #     bottomLeftCornerOfText = (x,y)
    #     cv2.putText(pseudo,type,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)


    plt.imshow(pseudo)
    for i,(type,col,x,y) in enumerate(zip(tt,body_colors,[pad,pad,t_h-pad,t_h-pad],[pad,2*pad,pad,2*pad])):
        x = 6
        y = i*6+6
        plt.text(x, y, type, fontsize=12,color = col)


    pseudo_net = np.zeros((target_image.shape[1],target_image.shape[2],3))

    affinity_colors = ['dodgerblue','yellow','purple','red','lime','orange','hotpink']
    for i,col in enumerate(affinity_colors):
        bright = target_pose[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        pseudo_net += color_im

    plt.subplot(2,3,3)
    plt.imshow(pseudo_net.clip(0,1))
    plt.title("affinity field")

    plt.subplot(2,3,2)
    plt.title("pixel targets, h: {} w: {}".format(pseudo_net.shape[0],pseudo_net.shape[1]))


    for i,(t,col) in enumerate(zip(["I --> E","I --> N","I --> T","E --> E","E --> T","E --> N","N --> T"],affinity_colors)):
        plt.subplot(4,4,9+i)
        bright = target_pose[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        plt.imshow( color_im/np.max(color_im) )
#         plt.imshow( color_im)
        
#         plt.imshow(bright)        
        plt.axis('off')
        plt.title(t)

        
    # ADD FINAL TOUCH 
    plt.subplot(4,4,16)
    show = np.copy(pseudo_net.clip(0,1))
    add_me = pseudo.clip(0,1)
    mask_me = np.any(add_me > .3,2)
    show[mask_me,:] = add_me[mask_me,:]
    plt.imshow(show)
        
    if save_fig:
        plt.savefig('cinema_training/trainframe_{}_.png'.format(np.random.uniform()))

    plt.show()
#%%

        
def plot_im_target_pseudo_ir(input_var,target_var,size = 10,save_fig = False):
    # def show_frame(input_var,target_var):
    # plt.imshow(input_var.data.cpu()[0,:,:,:].numpy())
    input_image = input_var.data.cpu()[0,0,:,:].numpy()

    target_stack = target_var.data.cpu()[0,:,:,:].numpy()

    target_image = target_stack[:4,...]
    target_pose = target_stack[4:,...]

    # = np.moveaxis(target[0,:,:,:].numpy() ,0,2)
    # test.shape
    # score_map = output.data.cpu()


    tt = ["implant","ears","noses",'tails']


    # show the tracking belief map
    Fig1 = plt.figure(figsize=(1.5*size,size))


    plt.subplot(2,3,1)
    plt.imshow(input_image,cmap='gray')
    plt.title("image space, h: {} w: {}".format(input_image.shape[0],input_image.shape[1]) )

    plt.subplot(2,3,2)
    # from matplotlib.pyplot import cm

    pseudo = np.zeros((target_image.shape[1],target_image.shape[2],3))

    body_colors =['dodgerblue','red','lime','orange']
    for i,col in enumerate(body_colors):
        bright = target_image[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        pseudo += color_im

    pseudo = np.clip(pseudo,0,1)

    # # Write some Text
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = .4
    # lineType = 0
    t_h,t_w = pseudo.shape[:2]
    pad = 10
    # for i,(type,col,x,y) in enumerate(zip(tt,body_colors,[pad,pad,t_h-pad,t_h-pad],[pad,2*pad,pad,2*pad])):
    # # do as in-place?
    #     rgb = Color(col).rgb
    #     fontColor = rgb
    #     bottomLeftCornerOfText = (10,i*10+20)
    #     bottomLeftCornerOfText = (x,y)
    #     cv2.putText(pseudo,type,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)


    plt.imshow(pseudo)
    for i,(type,col,x,y) in enumerate(zip(tt,body_colors,[pad,pad,t_h-pad,t_h-pad],[pad,2*pad,pad,2*pad])):
        x = 6
        y = i*6+6
        plt.text(x, y, type, fontsize=12,color = col)


    pseudo_net = np.zeros((target_image.shape[1],target_image.shape[2],3))

    affinity_colors = ['dodgerblue','yellow','purple','red','lime','orange','hotpink']
    for i,col in enumerate(affinity_colors):
        bright = target_pose[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        pseudo_net += color_im

    plt.subplot(2,3,3)
    plt.imshow(pseudo_net.clip(0,1))
    plt.title("affinity field")

    plt.subplot(2,3,2)
    plt.title("pixel targets, h: {} w: {}".format(pseudo_net.shape[0],pseudo_net.shape[1]))


    for i,(t,col) in enumerate(zip(["I --> E","I --> N","I --> T","E --> E","E --> T","E --> N","N --> T"],affinity_colors)):
        plt.subplot(4,4,9+i)
        bright = target_pose[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        plt.imshow( color_im/np.max(color_im) )
#         plt.imshow( color_im)
        
#         plt.imshow(bright)        
        plt.axis('off')
        plt.title(t)

        
    # ADD FINAL TOUCH 
    plt.subplot(4,4,16)
    show = np.copy(pseudo_net.clip(0,1))
    add_me = pseudo.clip(0,1)
    mask_me = np.any(add_me > .3,2)
    show[mask_me,:] = add_me[mask_me,:]
    plt.imshow(show)
        
    if save_fig:
        plt.savefig('cinema_training/trainframe_{}_.png'.format(np.random.uniform()))

    plt.show()
#%%


        
def convet_to_pseudo(target_var):
    # def show_frame(input_var,target_var):
    # plt.imshow(input_var.data.cpu()[0,:,:,:].numpy())

    target_stack = target_var.data.cpu()[0,:,:,:].numpy()

    target_image = target_stack[:4,...]
    target_pose = target_stack[4:,...]

    # = np.moveaxis(target[0,:,:,:].numpy() ,0,2)
    # test.shape
    # score_map = output.data.cpu()


    tt = ["implant","ears","noses",'tails']

    pseudo = np.zeros((target_image.shape[1],target_image.shape[2],3))

    body_colors =['dodgerblue','red','lime','orange']
    for i,col in enumerate(body_colors):
        bright = target_image[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        pseudo += color_im

    pseudo = np.clip(pseudo,0,1)

    pseudo_net = np.zeros((target_image.shape[1],target_image.shape[2],3))

    affinity_colors = ['dodgerblue','yellow','purple','red','lime','orange','hotpink']
    for i,col in enumerate(affinity_colors):
        bright = target_pose[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        pseudo_net += color_im

    pseudo_net = np.clip(pseudo_net,0,1)        
        
    show = np.copy(pseudo_net.clip(0,1))
    add_me = pseudo.clip(0,1)
    mask_me = np.any(add_me > .3,2)
    show[mask_me,:] = add_me[mask_me,:]

    return pseudo, pseudo_net,show






def plot_and_dump_im_target_pseudo(input_var,target_var,size = 10,save_fig = False):
    # def show_frame(input_var,target_var):
    # plt.imshow(input_var.data.cpu()[0,:,:,:].numpy())
    input_image = input_var.data.cpu()[0,:3,:,:].numpy()
    input_image = np.moveaxis(input_image,0,2)

    target_stack = target_var.data.cpu()[0,:,:,:].numpy()

    target_image = target_stack[:4,...]
    target_pose = target_stack[4:,...]

    # = np.moveaxis(target[0,:,:,:].numpy() ,0,2)
    # test.shape
    # score_map = output.data.cpu()


    tt = ["implant","ears","noses",'tails']


    # show the tracking belief map
    Fig1 = plt.figure(figsize=(1.5*size,size))


    plt.subplot(2,3,1)
    plt.imshow(input_image[:,:,[2,1,0]])
    
    figure_dump_folder = '/home/chrelli/git/3d_sandbox/mouseposev0p2/figure_raw_pics/figure_3'
    cv2.imwrite(figure_dump_folder+'/train/dump_im'+'.png',input_image*255)
    
    plt.title("image space, h: {} w: {}".format(input_image.shape[0],input_image.shape[1]) )

    plt.subplot(2,3,2)
    # from matplotlib.pyplot import cm

    pseudo = np.zeros((target_image.shape[1],target_image.shape[2],3))

    body_colors =['dodgerblue','red','lime','orange']
    for i,col in enumerate(body_colors):
        bright = target_image[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        pseudo += color_im

    pseudo = np.clip(pseudo,0,1)

    # # Write some Text
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = .4
    # lineType = 0
    t_h,t_w = pseudo.shape[:2]
    pad = 10
    # for i,(type,col,x,y) in enumerate(zip(tt,body_colors,[pad,pad,t_h-pad,t_h-pad],[pad,2*pad,pad,2*pad])):
    # # do as in-place?
    #     rgb = Color(col).rgb
    #     fontColor = rgb
    #     bottomLeftCornerOfText = (10,i*10+20)
    #     bottomLeftCornerOfText = (x,y)
    #     cv2.putText(pseudo,type,bottomLeftCornerOfText,font,fontScale,fontColor,lineType)


    plt.imshow(pseudo)
    cv2.imwrite(figure_dump_folder+'/train/dump_pseudo_targets'+'.png',pseudo[:,:,[2,1,0]]*255)
    
    for i,(type,col,x,y) in enumerate(zip(tt,body_colors,[pad,pad,t_h-pad,t_h-pad],[pad,2*pad,pad,2*pad])):
        x = 6
        y = i*6+6
        plt.text(x, y, type, fontsize=12,color = col)


    pseudo_net = np.zeros((target_image.shape[1],target_image.shape[2],3))

    affinity_colors = ['dodgerblue','yellow','purple','red','lime','orange','hotpink']
    for i,col in enumerate(affinity_colors):
        bright = target_pose[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        pseudo_net += color_im

    plt.subplot(2,3,3)
    plt.imshow(pseudo_net.clip(0,1))
    dddump = pseudo_net[:,:,[2,1,0]].clip(0,1)
    cv2.imwrite(figure_dump_folder+'/train/dump_pseudo_pafs'+'.png',dddump/np.max(dddump)*255)

    plt.title("affinity field")

    plt.subplot(2,3,2)
    plt.title("pixel targets, h: {} w: {}".format(pseudo_net.shape[0],pseudo_net.shape[1]))


    for i,(t,col) in enumerate(zip(["I --> E","I --> N","I --> T","E --> E","E --> T","E --> N","N --> T"],affinity_colors)):
        plt.subplot(4,4,9+i)
        bright = target_pose[i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        plt.imshow( color_im/np.max(color_im) )
#         plt.imshow( color_im)
        cv2.imwrite(figure_dump_folder+'/train/dump_pafs'+t+'.png',color_im[:,:,[2,1,0]]/np.max(color_im)*255)

#         plt.imshow(bright)        
        plt.axis('off')
        plt.title(t)

    if save_fig:
        plt.savefig('cinema_training/trainframe_{}_.png'.format(np.random.uniform()))

    plt.show()

#%%

def plot_ito_pseudo(input_var,target_var,output,size = 10):
    # def show_frame(input_var,target_var):
    # plt.imshow(input_var.data.cpu()[0,:,:,:].numpy())
    input_image = input_var.data.cpu()[0,:3,:,:].numpy()
    input_image = np.moveaxis(input_image,0,2)

    target_stack = target_var.data.cpu()[0,:,:,:].numpy()

    # clip the target to 1!
    target_stack = np.clip(target_stack,0,1)
    
    target_image = target_stack[:4,...]
    target_pose = target_stack[4:,...]

    # = np.moveaxis(target[0,:,:,:].numpy() ,0,2)
    # test.shape
    score_map = output[-1].data.cpu().numpy()

    tt = ["implant","ears","noses",'tails']

    # show the tracking belief map
    Fig1 = plt.figure(figsize=(1.5*size,size))

    plt.subplot(3,3,1)
    plt.title('image space')
    plt.imshow(input_image[:,:,[2,1,0]])

    plt.subplot(3,3,2)
    # from matplotlib.pyplot import cm

    body_colors =['dodgerblue','red','lime','orange']

    def color_target_points(target_image,body_colors):
        pseudo = np.zeros((target_image.shape[1],target_image.shape[2],3))
        for i,col in enumerate(body_colors):
            bright = target_image[i,:,:]
            rgb = Color(col).rgb
            color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
            pseudo += color_im
        pseudo = np.clip(pseudo,0,1)
        return pseudo

    pseudo = color_target_points(target_image,body_colors)

    t_h,t_w = pseudo.shape[:2]
    pad = 10

    plt.imshow(pseudo)
    for i,(type,col,x,y) in enumerate(zip(tt,body_colors,[pad,pad,t_h-pad,t_h-pad],[pad,2*pad,pad,2*pad])):
        x = 10
        y = i*10+10
        plt.text(x, y, type, fontsize=18,color = col)

    plt.title("pixel targets")

    affinity_colors = ['dodgerblue','yellow','purple','red','lime','orange','hotpink']

    def color_target_lines(target_pose,affinity_colors):
        pseudo_net = np.zeros((target_pose.shape[1],target_pose.shape[2],3))
        for i,col in enumerate(affinity_colors):
            bright = target_pose[i,:,:]
            rgb = Color(col).rgb
            color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
            pseudo_net += color_im
        pseudo_net = pseudo_net.clip(0,1)
        return pseudo_net


    plt.subplot(3,3,3)
    pseudo_net = color_target_lines(target_pose,affinity_colors)
    plt.imshow(pseudo_net)
    plt.title("affinity field")

    plt.subplot(3,3,5)
    pseudo_belief = color_target_points(score_map[0,:4,:,:],body_colors)
    plt.imshow(pseudo_belief)
    plt.title("network belief")

    plt.subplot(3,3,6)
    pseudo_belief = color_target_lines(score_map[0,4:,:,:],affinity_colors)
    plt.imshow(pseudo_belief)
    plt.title("network belief")

    pseudo = np.zeros((target_image.shape[1],target_image.shape[2],3))
    for i,col in enumerate(body_colors):
        plt.subplot(3,4,9+i)
        bright = score_map[0,i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        color_im = color_im.clip(0,1)
        plt.imshow(color_im)

    return Fig1


def plot_ito_pseudo_ir(input_var,target_var,output,size = 10):
    # def show_frame(input_var,target_var):
    # plt.imshow(input_var.data.cpu()[0,:,:,:].numpy())
    input_image = input_var.data.cpu()[0,0,:,:].numpy()

    target_stack = target_var.data.cpu()[0,:,:,:].numpy()

    # clip the target to 1!
    target_stack = np.clip(target_stack,0,1)
    
    target_image = target_stack[:4,...]
    target_pose = target_stack[4:,...]

    # = np.moveaxis(target[0,:,:,:].numpy() ,0,2)
    # test.shape
    score_map = output[-1].data.cpu().numpy()

    tt = ["implant","ears","noses",'tails']

    # show the tracking belief map
    Fig1 = plt.figure(figsize=(1.5*size,size))

    plt.subplot(3,3,1)
    plt.title('image space')
    plt.imshow(input_image)

    plt.subplot(3,3,2)
    # from matplotlib.pyplot import cm

    body_colors =['dodgerblue','red','lime','orange']

    def color_target_points(target_image,body_colors):
        pseudo = np.zeros((target_image.shape[1],target_image.shape[2],3))
        for i,col in enumerate(body_colors):
            bright = target_image[i,:,:]
            rgb = Color(col).rgb
            color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
            pseudo += color_im
        pseudo = np.clip(pseudo,0,1)
        return pseudo

    pseudo = color_target_points(target_image,body_colors)

    t_h,t_w = pseudo.shape[:2]
    pad = 10

    plt.imshow(pseudo)
    for i,(type,col,x,y) in enumerate(zip(tt,body_colors,[pad,pad,t_h-pad,t_h-pad],[pad,2*pad,pad,2*pad])):
        x = 10
        y = i*10+10
        plt.text(x, y, type, fontsize=18,color = col)

    plt.title("pixel targets")

    affinity_colors = ['dodgerblue','yellow','purple','red','lime','orange','hotpink']

    def color_target_lines(target_pose,affinity_colors):
        pseudo_net = np.zeros((target_pose.shape[1],target_pose.shape[2],3))
        for i,col in enumerate(affinity_colors):
            bright = target_pose[i,:,:]
            rgb = Color(col).rgb
            color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
            pseudo_net += color_im
        pseudo_net = pseudo_net.clip(0,1)
        return pseudo_net


    plt.subplot(3,3,3)
    pseudo_net = color_target_lines(target_pose,affinity_colors)
    plt.imshow(pseudo_net)
    plt.title("affinity field")

    plt.subplot(3,3,5)
    pseudo_belief = color_target_points(score_map[0,:4,:,:],body_colors)
    plt.imshow(pseudo_belief)
    plt.title("network belief")

    plt.subplot(3,3,6)
    pseudo_belief = color_target_lines(score_map[0,4:,:,:],affinity_colors)
    plt.imshow(pseudo_belief)
    plt.title("network belief")

    pseudo = np.zeros((target_image.shape[1],target_image.shape[2],3))
    for i,col in enumerate(body_colors):
        plt.subplot(3,4,9+i)
        bright = score_map[0,i,:,:]
        rgb = Color(col).rgb
        color_im = bright[:,:,np.newaxis] * np.asarray(rgb)[np.newaxis,np.newaxis,:]
        color_im = color_im.clip(0,1)
        plt.imshow(color_im)

    return Fig1


# # EXAMPLE OF AUGMENTATION

# index = 5

# geometry = pickle.load( open( tracking_folder+'/geometry.pkl', "rb" ) ) 
# depth_scale = geometry['d_cam_params'][3][4]

# xy = h5_file['annotations'][index]
# c_image = h5_file['c_images'][index]
# dac_image = h5_file['dac_images'][index]

# # images = im[:3,:,:].astype('float32')
# # images = np.moveaxis(images,0,2)[np.newaxis,:,:,:]

# images = c_image[np.newaxis,:,:,[2,1,0]]

# import imgaug.augmenters as iaa

# seq = iaa.Sequential([
# #     iaa.Crop(px=(0, 100)), # crop images from each side by 0 to 16px (randomly chosen)
#     iaa.CropAndPad(percent=(-0.05, 0.15), sample_independently=False),
#     iaa.Fliplr(0.5), # horizontally flip 50% of the images
#     iaa.Sometimes(.2, iaa.GaussianBlur(sigma=(0, 1.5)) ), # blur images with a sigma of 0 to 3.0
#     iaa.Sometimes( 1, iaa.Dropout(p = (0,0.2)) ),
#     iaa.Affine(rotate=(-30, 30)),
#     iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
# ])

# for _ in range(5):
#     images_aug, xy_aug_list = seq(images = images, keypoints=[xy])
#     xy_aug = xy_aug_list[0]
#     plt.figure(figsize=(15,15))
    
#     plt.subplot(2,2,1)
#     plt.imshow(images[0,...])
#     plt.plot(xy[:,0],xy[:,1],'or')

#     plt.subplot(2,2,2)
#     plt.imshow(images_aug[0,...])
#     plt.plot(xy_aug[:,0],xy_aug[:,1],'or')
    
#     plt.show()

# plt.figure(figsize = (20,20))
# st = 620
# for i,index in enumerate(range(st,st+100)):
#     plt.subplot(10,10,1+i)
#     c_image = h5_file['c_images'][index]
#     # blank_image = np.zeros_like(c_image)
#     plt.imshow(c_image[:,:,[2,1,0]])
#     plt.title(index)
# plt.show()
# #     cv2.imshow('hm',c_image[:,:,:])
# #     cv2.waitKey(500)



# index = 10
# xy = h5_file['annotations'][index]
# c_image = h5_file['c_images'][index]
# dac_image = h5_file['dac_images'][index]

# for index in range(10):
#     c_image = h5_file['c_images'][index]
#     # blank_image = np.zeros_like(c_image)
    
#     cv2.imshow('hm',c_image[:,:,:])
#     cv2.waitKey(500)

# cv2.destroyAllWindows()