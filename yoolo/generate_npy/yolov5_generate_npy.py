# coding: utf-8
import torch
import numpy as np
import argparse

def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grids = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
    return grids

def Yolov5_generate_npy(input_h, input_w) :  
    name = ['80','40','20']
    ny, nx = int(input_h // pow(2,3)), int(input_w // pow(2,3))
    for i in range(3):
        ny_i, nx_i = int(ny // pow(2,i)), int(nx // pow(2,i))
        grid = make_grid(nx_i, ny_i) 
        grid = grid.numpy() 
        file_name = name[i]+'_'+str(input_w)+'x'+str(input_h)+'.npy' 
        np.save(file_name,grid) 
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-h', type=int, default=640, help='input height')
    parser.add_argument('--input-w', type=int, default=640, help='input width')
    args = parser.parse_args()
    
    input_h, input_w = args.input_h, args.input_w
    Yolov5_generate_npy(input_h, input_w)
