from __future__ import print_function
import torch
import math
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import renderer
import time
import sys, os
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import numpy as np
import cv2
from math import exp
from PIL import Image
from scipy.spatial.transform import Rotation as R
import random
from torch.optim import lr_scheduler
import trimesh
import mesh2sdf
from Loss import MS_SSIM,SSIM
import os
import yaml
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#--------------------------------------------------- Unet ---------------------------------------------
class Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(Unet, self).__init__()
        
        # Encoder
        self.inc = self._block(n_channels, 64)
        self.down1 = self._down(64, 128)
        self.down2 = self._down(128, 256)
        self.down3 = self._down(256, 512)
        self.down4 = self._down(512, 1024)
        
        # Decoder
        self.up1 = self._up(1024, 512)
        self.up2 = self._up(512, 256)
        self.up3 = self._up(256, 128)
        self.up4 = self._up(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, 1)
        
        # Predefine all blocks
        self.block1 = self._block(1024, 512)
        self.block2 = self._block(512, 256)
        self.block3 = self._block(256, 128)
        self.block4 = self._block(128, 64)

    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _down(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            self._block(in_ch, out_ch)
        )

    def _up(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.block1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.block2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.block3(x)
        
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.block4(x)
        
        return self.outc(x)
        
        

class SDFGrid(nn.Module):
    def __init__(self, bounding_box_min, bounding_box_max, grid_res, cuda, filepath = None):
        super().__init__()
        self.bounding_box_min = bounding_box_min
        self.bounding_box_max = bounding_box_max
        self.grid_res = grid_res
        self.voxel_size = (self.bounding_box_max - self.bounding_box_min) / (self.grid_res - 1)

        self.relu=nn.ReLU()
        if filepath:
            grid = self.read_grid(filepath)   
        else:
            grid = self.init_grid()
            
        if cuda:
            self.grid = grid.cuda()
        else:
            self.grid = grid
            
    def init_grid(self):
        linear_space = torch.linspace(self.bounding_box_min, self.bounding_box_max, self.grid_res)
        x_dim = linear_space.view(-1, 1).repeat(self.grid_res, 1, self.grid_res)
        y_dim = linear_space.view(1, -1).repeat(self.grid_res, self.grid_res, 1)
        z_dim = linear_space.view(-1, 1, 1).repeat(1, self.grid_res, self.grid_res)
        grid = torch.sqrt(x_dim * x_dim + y_dim * y_dim + z_dim * z_dim) - 0.5 * self.bounding_box_max
        return grid
                
    def read_grid(self, filename):
        mesh = trimesh.load(filename, force='mesh')
        vertices = mesh.vertices

        vertices = vertices / self.bounding_box_max
        
        sdf, mesh = mesh2sdf.compute(vertices, mesh.faces, self.grid_res, fix=True, level=2/self.grid_res, return_mesh=True)
        sdf = Tensor(sdf*self.bounding_box_max)
        return sdf

    # normal of the surface point
    def get_grid_normal(self):
        n_x = self.grid_res - 1
        n_y = self.grid_res - 1
        n_z = self.grid_res - 1

        # x-axis normal vectors
        X_1 = torch.cat((self.grid[1:, :, :], ((3 * self.grid[n_x, :, :] - self.grid[n_x - 2, :, :]) / 2).unsqueeze_(0)), 0)
        X_2 = torch.cat((((3 * self.grid[0, :, :] - self.grid[2, :, :]) / 2).unsqueeze_(0), self.grid[:n_x, :, :]), 0)
        grid_normal_x = (X_1 - X_2) / (2 * self.voxel_size)

        # y-axis normal vectors
        Y_1 = torch.cat((self.grid[:, 1:, :], ((3 * self.grid[:, n_y, :] - self.grid[:, n_y - 2, :]) / 2).unsqueeze_(1)), 1)
        Y_2 = torch.cat((((3 * self.grid[:, 0, :] - self.grid[:, 2, :]) / 2).unsqueeze_(1), self.grid[:, :n_y, :]), 1)
        grid_normal_y = (Y_1 - Y_2) / (2 * self.voxel_size)

        # z-axis normal vectors
        Z_1 = torch.cat((self.grid[:, :, 1:], ((3 * self.grid[:, :, n_z] - self.grid[:, :, n_z - 2]) / 2).unsqueeze_(2)), 2)
        Z_2 = torch.cat((((3 * self.grid[:, :, 0] - self.grid[:, :, 2]) / 2).unsqueeze_(2), self.grid[:, :, :n_z]), 2)
        grid_normal_z = (Z_1 - Z_2) / (2 * self.voxel_size)
        return [grid_normal_x, grid_normal_y, grid_normal_z]


    # intersection normal
    def get_intersection_normal(self, intersection_grid_normal, intersection_pos, voxel_min_point):
        tx = (intersection_pos[:, :, 0] - voxel_min_point[:, :, 0]) / self.voxel_size
        ty = (intersection_pos[:, :, 1] - voxel_min_point[:, :, 1]) / self.voxel_size
        tz = (intersection_pos[:, :, 2] - voxel_min_point[:, :, 2]) / self.voxel_size

        intersection_normal = (1 - tz) * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, 0] \
                              + tz * (1 - ty) * (1 - tx) * intersection_grid_normal[:, :, 1] \
                              + (1 - tz) * ty * (1 - tx) * intersection_grid_normal[:, :, 2] \
                              + tz * ty * (1 - tx) * intersection_grid_normal[:, :, 3] \
                              + (1 - tz) * (1 - ty) * tx * intersection_grid_normal[:, :, 4] \
                              + tz * (1 - ty) * tx * intersection_grid_normal[:, :, 5] \
                              + (1 - tz) * ty * tx * intersection_grid_normal[:, :, 6] \
                              + tz * ty * tx * intersection_grid_normal[:, :, 7]

        return intersection_normal


    # generate shading image and shape image
    def generate_nlpl(self, width, height, dw, dh, r, d, point_max):
        # grid normal
        [grid_normal_x, grid_normal_y, grid_normal_z] = self.get_grid_normal()

        # MSPS: get surface points
        step_size = self.voxel_size
        w_h_3 = torch.zeros(width, height, 3).cuda()
        w_h_12 = torch.zeros(width, height, point_max).cuda()
        outputs = renderer.ray_matching(w_h_12, w_h_3, self.grid, width, height, self.bounding_box_min,
                                        self.bounding_box_max, self.grid_res, dw, dh, r[0], r[1], r[2], d[0], d[1],
                                        d[2], point_max, step_size)
        voxel_pos = outputs[2]
        intersection_pos = outputs[3]
        #spc=torch.sum(intersection_pos!=-1,2)


        # (256, 256, point_max) -> (256, 256*point_max/3, 3)
        cat_voxel_pos = voxel_pos[:, :, 0:3]
        cat_intersection_pos = intersection_pos[:, :, 0:3]
        for i in range(3, point_max, 3):
            cat_voxel_pos = torch.cat((cat_voxel_pos, voxel_pos[:, :, i:i + 3]), 1)  
            cat_intersection_pos = torch.cat((cat_intersection_pos, intersection_pos[:, :, i:i + 3]), 1)

        # Make the pixels with no intersections with rays be 0
        mask = (cat_voxel_pos[:, :, 0] != -1).type(Tensor)

        # Get the indices of the minimum point of the intersecting voxels
        x = cat_voxel_pos[:, :, 0].type(torch.cuda.LongTensor)
        y = cat_voxel_pos[:, :, 1].type(torch.cuda.LongTensor)
        z = cat_voxel_pos[:, :, 2].type(torch.cuda.LongTensor)
        x[x == -1] = 0
        y[y == -1] = 0
        z[z == -1] = 0
        
        
        #---------------------------
        grid_temp=self.relu(-self.grid)
        
        gridx=torch.index_select(grid_temp.view(-1), 0,
                                z.view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        
        
        gridx[mask == 0] = 0
        im = gridx[:, 0:width]
        for i in range(3, point_max, 3):
            im = im + gridx[:, int(i / 3) * width:int(i / 3) * width + width]   # shape image
        
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 416, 416)
        cv2.imshow('output',np.array((im*255).cpu().detach().numpy()).astype(np.uint8))
        cv2.waitKey(1)

        
        # This line is equivalent to grid_normal_x[x,y,z]
        x1 = torch.index_select(grid_normal_x.view(-1), 0,
                                z.view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        x2 = torch.index_select(grid_normal_x.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        x3 = torch.index_select(grid_normal_x.view(-1), 0,
                                z.view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        x4 = torch.index_select(grid_normal_x.view(-1), 0, (z + 1).view(-1) + self.grid_res * (y + 1).view(
            -1) + self.grid_res * self.grid_res * x.view(-1)).view(x.shape).unsqueeze_(2)
        x5 = torch.index_select(grid_normal_x.view(-1), 0,
                                z.view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * (x + 1).view(
                                    -1)).view(x.shape).unsqueeze_(2)
        x6 = torch.index_select(grid_normal_x.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        x7 = torch.index_select(grid_normal_x.view(-1), 0,
                                z.view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        x8 = torch.index_select(grid_normal_x.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        intersection_grid_normal_x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8), 2) + (
                    1 - mask.view(height, int(width * point_max / 3), 1).repeat(1, 1, 8))

        # Get the y-axis of normal vectors for the 8 points of the intersecting voxel
        y1 = torch.index_select(grid_normal_y.view(-1), 0,
                                z.view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        y2 = torch.index_select(grid_normal_y.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        y3 = torch.index_select(grid_normal_y.view(-1), 0,
                                z.view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        y4 = torch.index_select(grid_normal_y.view(-1), 0, (z + 1).view(-1) + self.grid_res * (y + 1).view(
            -1) + self.grid_res * self.grid_res * x.view(-1)).view(x.shape).unsqueeze_(2)
        y5 = torch.index_select(grid_normal_y.view(-1), 0,
                                z.view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * (x + 1).view(
                                    -1)).view(x.shape).unsqueeze_(2)
        y6 = torch.index_select(grid_normal_y.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        y7 = torch.index_select(grid_normal_y.view(-1), 0,
                                z.view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        y8 = torch.index_select(grid_normal_y.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        intersection_grid_normal_y = torch.cat((y1, y2, y3, y4, y5, y6, y7, y8), 2) + (
                    1 - mask.view(height, int(width * point_max / 3), 1).repeat(1, 1, 8))

        # Get the z-axis of normal vectors for the 8 points of the intersecting voxel
        z1 = torch.index_select(grid_normal_z.view(-1), 0,
                                z.view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        z2 = torch.index_select(grid_normal_z.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        z3 = torch.index_select(grid_normal_z.view(-1), 0,
                                z.view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * x.view(
                                    -1)).view(x.shape).unsqueeze_(2)
        z4 = torch.index_select(grid_normal_z.view(-1), 0, (z + 1).view(-1) + self.grid_res * (y + 1).view(
            -1) + self.grid_res * self.grid_res * x.view(-1)).view(x.shape).unsqueeze_(2)
        z5 = torch.index_select(grid_normal_z.view(-1), 0,
                                z.view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * (x + 1).view(
                                    -1)).view(x.shape).unsqueeze_(2)
        z6 = torch.index_select(grid_normal_z.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * y.view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        z7 = torch.index_select(grid_normal_z.view(-1), 0,
                                z.view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        z8 = torch.index_select(grid_normal_z.view(-1), 0,
                                (z + 1).view(-1) + self.grid_res * (y + 1).view(-1) + self.grid_res * self.grid_res * (
                                            x + 1).view(-1)).view(x.shape).unsqueeze_(2)
        intersection_grid_normal_z = torch.cat((z1, z2, z3, z4, z5, z6, z7, z8), 2) + (
                    1 - mask.view(height, int(width * point_max / 3), 1).repeat(1, 1, 8))

        # voxel -> point
        cat_voxel_min_point = Tensor(
            [self.bounding_box_min, self.bounding_box_min, self.bounding_box_min]) + cat_voxel_pos * self.voxel_size


        # Compute the normal vectors for the intersecting points
        intersection_normal_x = self.get_intersection_normal(intersection_grid_normal_x, cat_intersection_pos,
                                                             cat_voxel_min_point)
        intersection_normal_y = self.get_intersection_normal(intersection_grid_normal_y, cat_intersection_pos,
                                                             cat_voxel_min_point)
        intersection_normal_z = self.get_intersection_normal(intersection_grid_normal_z, cat_intersection_pos,
                                                             cat_voxel_min_point)

        # Put all the xyz-axis of the normal vectors into a single matrix
        intersection_normal_x_resize = intersection_normal_x.unsqueeze_(2)
        intersection_normal_y_resize = intersection_normal_y.unsqueeze_(2)
        intersection_normal_z_resize = intersection_normal_z.unsqueeze_(2)
        intersection_normal = torch.cat(
            (intersection_normal_x_resize, intersection_normal_y_resize, intersection_normal_z_resize), 2)
        intersection_normal = intersection_normal / torch.unsqueeze(torch.norm(intersection_normal, p=2, dim=2),
                                                                    2).repeat(1, 1, 3)

        # generate shading image 
        light_direction = -r.repeat(height, int(width * point_max / 3), 1)
        l_dot_n = torch.sum(light_direction * intersection_normal, 2).unsqueeze_(2)
        shading = torch.max(l_dot_n, Tensor(height, int(width * point_max / 3), 1).fill_(0))[:, :, 0]
        ln = shading * mask
        ln[mask == 0] = 0
        
        img = ln[:, 0:width]
        for i in range(3, point_max, 3):
            img = img + ln[:, int(i / 3) * width:int(i / 3) * width + width]
        
        max_num=torch.max(img)
        max_num=max_num.detach()        
        img = img/max_num        
        out=img

        cv2.namedWindow('output1', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output1', 416, 416)
        cv2.imshow('output1',np.array((out*255).cpu().detach().numpy()).astype(np.uint8))
        cv2.waitKey(1)

        return out.unsqueeze_(2),mask,im

    # loss
    def loss(self):       
        [grid_normal_x, grid_normal_y, grid_normal_z] = self.get_grid_normal()  
        
        eik_loss = torch.mean(torch.pow(torch.sqrt(torch.pow(grid_normal_x,2)+torch.pow(grid_normal_y,2)+torch.pow(grid_normal_z,2))-1,2))
        
        conv_input = self.grid.unsqueeze(0).unsqueeze(0)
        conv_filter = torch.cuda.FloatTensor([[[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                                [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]])
        curv_loss = torch.mean(F.conv3d(conv_input, conv_filter) ** 2)
        
        return eik_loss, curv_loss
        


def reconstruction(cfg):
    width = cfg['Image']['width']
    height = cfg['Image']['height']
    dh = cfg['Image']['dh']
    dw = cfg['Image']['dw']
    img_type = cfg['Image']['type']
    
    bounding_box_min = cfg['SDF']['bounding_box_min']
    bounding_box_max = cfg['SDF']['bounding_box_max']  
    voxel_res_list = cfg['SDF']['voxel_res_list']
    voxel_res_end = voxel_res_list[-1]-1
    
    init_model_path = cfg['Train']['init_model_path']
    RenderNet_weight_path = cfg['Train']['RenderNet_weight_path']
    epochs_list = cfg['Train']['epochs_list']
    lrate_list = cfg['Train']['lrate_list']
    
    eik_w = cfg['Train']['eik_weight']
    curv_w = cfg['Train']['curv_weight']
    Ms_s_weight = cfg['Train']['Ms_s_weight']
    
    angle_path = cfg['Train']['Input']['angle_path']
    image_path = cfg['Train']['Input']['image_path']
    
    
    # filename of the results  
    dir_name = cfg['Train']['save_path']
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    grid_res = voxel_res_list.pop(0)
    point_max = 24

    # Input data
    view_list = []
    image_target = []  
    record = np.load(angle_path)

    for i in range(len(record)):
        view_list.append(Tensor(record[i,:]))
        
        img_path = image_path + str(i+1) + '.' + img_type
        image = cv2.imread(img_path)   
        image = Tensor(np.transpose(image, (2, 0, 1)))/255
        image_target.append(image)
    
    # Rendering network
    net=Unet(1,3)
    net.load_state_dict(torch.load(RenderNet_weight_path)) 
    net.cuda()
    net.eval()
    for name, param in net.named_parameters():
        param.requires_grad = False

    # SDF grid init
    sdf_init = SDFGrid(bounding_box_min, bounding_box_max, grid_res, cuda, filepath= init_model_path)    


    # loss
    l1_loss=nn.L1Loss(reduction='mean')
    msssim_loss=MS_SSIM()
    
    
    # reconstruction
    start_time = time.time()
    while sdf_init.grid_res <= voxel_res_end:
        sdf_init.grid.requires_grad = True
        
        # optimizer
        params = [sdf_init.grid]
        lrate = lrate_list.pop(0)  
        optimizer = torch.optim.NAdam(params, lr=lrate, betas=(0.9, 0.999), eps=1e-04, weight_decay=0, momentum_decay=0.04)

        train_epochs_loss = []
        epochs = epochs_list.pop(0)
          
        for epoch in range(epochs):
            train_epoch_loss = []
            
            for cam in range(len(view_list)):
                optimizer.zero_grad()
                # view
                r = view_list[cam][0:3]
                d = view_list[cam][3:6]
                
                # render
                image_nlpl,mask,im = sdf_init.generate_nlpl(width, height, dw, dh, r, d, point_max)    
                image_new = image_nlpl.permute(2, 0, 1).unsqueeze(0)    # Shading image
                image_initial = net(image_new).squeeze()                # rendered ISAR image 
                    
                # loss
                eik_loss, curv_loss = sdf_init.loss()
                  
                if epoch<500:
                  mss_w=Ms_s_weight*(1+(500-epoch)/500)
                else:
                  mss_w=Ms_s_weight

                loss2=l1_loss(image_initial.unsqueeze(0), image_target[cam].unsqueeze(0))*1
                loss3=(1-msssim_loss(image_initial.unsqueeze(0),image_target[cam].unsqueeze(0)))*1   
                loss4=(1-msssim_loss(image_target[cam].unsqueeze(0),im.permute(2, 0, 1).repeat(1,3,1,1)))*mss_w             
                loss = eik_w*eik_loss + loss2 + loss4 + loss3 + curv_w*curv_loss
                
                # optimize
                loss.backward()
                optimizer.step()
                
                # print
                train_epoch_loss.append(loss.item())
                if cam%(len(view_list)//4)==0:
                    print("epoch={}/{},{}/{} of train, loss={:.4f}, eik_loss={:.4f}, curv_loss={:.4f}, l1={:.4f}, MS_r={:.4f}, MS_s={:.4f}".format(epoch, epochs, cam, len(view_list),loss.item(), eik_w*eik_loss.item(), curv_w*curv_loss.item(), loss2.item(),loss3.item(),loss4.item()))
                    
            train_epochs_loss.append(np.average(train_epoch_loss))
            print(np.average(train_epoch_loss).item())   

            if epoch%100==0:
                with open(dir_name + str(sdf_init.grid_res)+"_"+str(epoch) + "sdf.pt", 'wb') as f:
                    torch.save(sdf_init.grid, f)
                
        # save grid_init
        with open(dir_name + str(sdf_init.grid_res) + "_best_sdf1.pt", 'wb') as f:
            torch.save(sdf_init.grid, f)

        # new grid_res
        grid_res_update = voxel_res_list.pop(0)
    
        grid_res_temp = sdf_init.grid_res
        while grid_res_temp < grid_res_update:
            grid_res_temp = grid_res_temp + 8
            sdf_init.grid = F.interpolate(sdf_init.grid.unsqueeze(0).unsqueeze(0),size=(grid_res_temp, grid_res_temp, grid_res_temp), mode='trilinear').squeeze()
    
        sdf_init.grid_res = grid_res_update
        sdf_init.voxel_size = (sdf_init.bounding_box_max - sdf_init.bounding_box_min) / (grid_res_update - 1)
        sdf_init.grid = sdf_init.grid.detach()
    

    print("Time:", time.time() - start_time)
    print("----- END -----")

      
if __name__ == "__main__":
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # device
    torch.backends.cudnn.benchmark = True
    cuda = True if torch.cuda.is_available() else False
    print(cuda)
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # config
    with open("config.yaml", "r") as config:
        cfg = yaml.safe_load(config)
    
    # method
    reconstruction(cfg)
    