#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import time
import gc



# 权重映射，A矩阵生成
def weight_map_conv(weight, bias, xbar_row, xbar_column, Rm_max, Rm_off):
    "weight,bias are tensor from saved model"
    Cout = weight.shape[0]
    weight = weight.reshape(Cout,-1)
    weight_mapped = weight_map_linear(weight, bias, xbar_row=xbar_row, xbar_column=xbar_column, Rm_max=Rm_max, Rm_off=Rm_off)
    return weight_mapped
    
#     #默认卷积核尺寸k*k小于xbar_row
#     if Cout <= xbar_column:
#         N_kernel_per_xbar = xbar_row//(kernel_row*kernel_column)
#         xbar_v_num = Cin//N_kernel_per_xbar
        
#     #xbar_row_sum:垂直方向需要xbar数量(考虑bias)    
#     xbar_row_sum = (weight.shape[0]-1)//xbar_row+1
#     xbar_column_sum = (weight.shape[1]-1)//xbar_column+1
#     #kernel_num:每个xbar中容纳核的个数
#     kernel_num = xbar_row//(kernel_row*kernel_column)
 
    
#     weight_mapped = torch.zeros([xbar_row_sum][xbar_column_sum][xbar_row][xbar_column])
    
#     for i in range(xbar_row_sum):
#         for j in range(xbar_column_sum):
#             if (i != xbar_row_sum-1) and (j != xbar_column_sum-1):
#                 weight_mapped[i][j][:][:] = weight[i*xbar_row:(i+1)*xbar][j*xbar_column:(j+1)*xbar_column]
#             elif (i == xbar_row_sum) and (j != xbar_column_sum-1):
                    
#             weight_mapped[i][j][kernel_num*kernel_row*kernel_column][]
    
#     weight = weight.reshape(w_map_r,w_map_c)
#     xbar_column_sum = w_map_c//xbar_column+1
#     xbar_row_sum = w_map_r//xbar_row+1

# def weight_map_linear(weight,bias,xbar_row=64,xbar_column=64,Rm_max=50e3,Rm_off=10e9):
#     "weight mapping for linear layers;weight_mapped[i,j,:,:] represents the position where this xbar locate in the whole weight matrix"
#     weight = torch.vstack((weight.T,bias))
#     xbar_v_num = (weight.shape[0]-1)//xbar_row+1
#     xbar_h_num = (weight.shape[1]-1)//xbar_column+1
#     weight_mapped = torch.zeros(xbar_v_num,xbar_h_num,xbar_row,xbar_column)
#     for i in range(xbar_v_num):
#         for j in range(xbar_h_num):
#             if (i != xbar_v_num-1) and (j != xbar_h_num-1):
#                 weight_mapped[i,j,:,:] = weight[i*xbar_row:(i+1)*xbar_row,j*xbar_column:(j+1)*xbar_column]
#             elif (i != xbar_v_num-1) and (j == xbar_h_num-1):
#                 weight_mapped[i,j,:,:(weight.shape[1]-j*xbar_column)] = weight[i*xbar_row:(i+1)*xbar_row,j*xbar_column:]
#                 weight_mapped[i,j,:,(weight.shape[1]-j*xbar_column):] = 1/Rm_max
#             elif (i == xbar_v_num-1) and (j != xbar_h_num-1):
#                 weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),:] = weight[i*xbar_row:,j*xbar_column:(j+1)*xbar_column]
#                 weight_mapped[i,j,(weight.shape[0]-i*xbar_row):,:] = 1/Rm_off
#             else:
#                 weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),:(weight.shape[1]-j*xbar_column)] = weight[i*xbar_row:,j*xbar_column:]
#                 weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),(weight.shape[1]-j*xbar_column):] = 1/Rm_max
#                 weight_mapped[i,j,(weight.shape[0]-i*xbar_row):,:] = 1/Rm_off
                

        
#     return weight_mapped


def weight_map_linear(weight,bias,xbar_row,xbar_column,Rm_max,Rm_off):
    "weight mapping for linear layers;weight_mapped[i,j,:,:] represents the position where this xbar locate in the whole weight matrix"
    weight = torch.vstack((weight.T,bias))
    xbar_v_num = (weight.shape[0]-1)//xbar_row+1
    xbar_h_num = (weight.shape[1]-1)//xbar_column+1
    weight_mapped = torch.zeros(xbar_v_num,xbar_h_num,xbar_row,xbar_column)
    for i in range(xbar_v_num):
        for j in range(xbar_h_num):
            if (i != xbar_v_num-1) and (j != xbar_h_num-1):
                weight_mapped[i,j,:,:] = weight[i*xbar_row:(i+1)*xbar_row,j*xbar_column:(j+1)*xbar_column]
            elif (i != xbar_v_num-1) and (j == xbar_h_num-1):
                weight_mapped[i,j,:,:(weight.shape[1]-j*xbar_column)] = weight[i*xbar_row:(i+1)*xbar_row,j*xbar_column:]
                weight_mapped[i,j,:,(weight.shape[1]-j*xbar_column):] = 1/Rm_max
            elif (i == xbar_v_num-1) and (j != xbar_h_num-1):
                weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),:] = weight[i*xbar_row:,j*xbar_column:(j+1)*xbar_column]
                weight_mapped[i,j,(weight.shape[0]-i*xbar_row):,:] = 1/Rm_max
            else:
                weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),:(weight.shape[1]-j*xbar_column)] = weight[i*xbar_row:,j*xbar_column:]
                weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),(weight.shape[1]-j*xbar_column):] = 1/Rm_max
                weight_mapped[i,j,(weight.shape[0]-i*xbar_row):,:] = 1/Rm_max
                        
    return weight_mapped

# In[4]:


def W_to_G(weight,bias,Rm_min,Rm_max,Rm_off,xbar_row,xbar_column,linear=True):
    "weight transfer to G_p and G_n,linear=True"
    weight = torch.where(weight>1,torch.tensor(1.),weight)
    weight = torch.where(weight<-1,torch.tensor(-1.),weight)
    bias = torch.where(bias>1,torch.tensor(1.),bias)
    bias = torch.where(bias<-1,torch.tensor(-1.),bias)
    
    G_p = torch.where(weight>=0,weight*(1/Rm_min-1/Rm_max)+1/Rm_max,torch.tensor(1/Rm_max))
    bias_p = torch.where(bias>=0,bias*(1/Rm_min-1/Rm_max)+1/Rm_max,torch.tensor(1/Rm_max))
    G_n = torch.where(weight<0,-weight*(1/Rm_min-1/Rm_max)+1/Rm_max,torch.tensor(1/Rm_max))
    bias_n = torch.where(bias<0,-bias*(1/Rm_min-1/Rm_max)+1/Rm_max,torch.tensor(1/Rm_max))
    
    #map G_p and G_n to xbar
    if linear:
        weight_mapped_p = weight_map_linear(weight=G_p, bias=bias_p, xbar_row=xbar_row, xbar_column=xbar_column, Rm_max=Rm_max, Rm_off=Rm_off)
        weight_mapped_n = weight_map_linear(weight=G_n, bias=bias_n, xbar_row=xbar_row, xbar_column=xbar_column, Rm_max=Rm_max, Rm_off=Rm_off)
        
    else:
        weight_mapped_p = weight_map_conv(weight=G_p, bias=bias_p, xbar_row=xbar_row, xbar_column=xbar_column, Rm_max=Rm_max, Rm_off=Rm_off)
        weight_mapped_n = weight_map_conv(weight=G_n, bias=bias_n, xbar_row=xbar_row, xbar_column=xbar_column, Rm_max=Rm_max, Rm_off=Rm_off)
    
    return weight_mapped_p,weight_mapped_n


# **weight_map_linear()函数输入的weight形如(Cout,Cin)**<br><br>
# **weight_map_conv()函数输入的weight形如(Cout,Cin,k,k)**

############################################ 双电源方案 ##############################################
def weight_map_conv_2source(weight_p, weight_n, bias_p, bias_n, xbar_row=64, xbar_column=64, Rm_max=50e3, Rm_off=10e9):
    "weight,bias are tensor from saved model"
    Cout = weight_p.shape[0]
    weight_p = weight_p.reshape(Cout,-1)
    weight_n = weight_n.reshape(Cout,-1)
    
    weight_mapped = weight_map_linear_2source(weight_p, weight_n, bias_p, bias_n, xbar_row=xbar_row, xbar_column=xbar_column, Rm_max=Rm_max, Rm_off=Rm_off)
    return weight_mapped


def weight_map_linear_2source(weight_p,weight_n,bias_p,bias_n,xbar_row=64,xbar_column=64,Rm_max=50e3,Rm_off=10e9):
    "weight mapping for linear layers;weight_mapped[i,j,:,:] represents the position where this xbar locate in the whole weight matrix"
    weight_p = torch.vstack((weight_p.T,bias_p))
    weight_n = torch.vstack((weight_n.T,bias_n))
    weight = torch.zeros(weight_p.shape[0]*2,weight_p.shape[1])
    for i in range(weight_p.shape[0]):
        weight[2*i,:] = weight_p[i,:]
        weight[2*i+1,:] = weight_n[i,:]
    
    xbar_v_num = (weight.shape[0]-1)//xbar_row+1
    xbar_h_num = (weight.shape[1]-1)//xbar_column+1
    weight_mapped = torch.zeros(xbar_v_num,xbar_h_num,xbar_row,xbar_column)
    for i in range(xbar_v_num):
        for j in range(xbar_h_num):
            if (i != xbar_v_num-1) and (j != xbar_h_num-1):
                weight_mapped[i,j,:,:] = weight[i*xbar_row:(i+1)*xbar_row,j*xbar_column:(j+1)*xbar_column]
            elif (i != xbar_v_num-1) and (j == xbar_h_num-1):
                weight_mapped[i,j,:,:(weight.shape[1]-j*xbar_column)] = weight[i*xbar_row:(i+1)*xbar_row,j*xbar_column:]
                weight_mapped[i,j,:,(weight.shape[1]-j*xbar_column):] = 1/Rm_max
            elif (i == xbar_v_num-1) and (j != xbar_h_num-1):
                weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),:] = weight[i*xbar_row:,j*xbar_column:(j+1)*xbar_column]
                weight_mapped[i,j,(weight.shape[0]-i*xbar_row):,:] = 1/Rm_off
            else:
                weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),:(weight.shape[1]-j*xbar_column)] = weight[i*xbar_row:,j*xbar_column:]
                weight_mapped[i,j,:(weight.shape[0]-i*xbar_row),(weight.shape[1]-j*xbar_column):] = 1/Rm_max
                weight_mapped[i,j,(weight.shape[0]-i*xbar_row):,:] = 1/Rm_off
                        
    return weight_mapped


def W_to_G_2source(weight,bias,Rm_min=10e3,Rm_max=50e3,Rm_off=10e9,xbar_row=64,xbar_column=64,linear=True):
    "weight transfer to G_p and G_n,linear=True"
    weight = torch.where(weight>1,torch.tensor(1.),weight)
    weight = torch.where(weight<-1,torch.tensor(-1.),weight)
    bias = torch.where(bias>1,torch.tensor(1.),bias)
    bias = torch.where(bias<-1,torch.tensor(-1.),bias)
    
    G_p = torch.where(weight>=0,weight*(1/Rm_min-1/Rm_max)+1/Rm_max,torch.tensor(1/Rm_max))
    bias_p = torch.where(bias>=0,bias*(1/Rm_min-1/Rm_max)+1/Rm_max,torch.tensor(1/Rm_max))
    G_n = torch.where(weight<0,-weight*(1/Rm_min-1/Rm_max)+1/Rm_max,torch.tensor(1/Rm_max))
    bias_n = torch.where(bias<0,-bias*(1/Rm_min-1/Rm_max)+1/Rm_max,torch.tensor(1/Rm_max))
    
    #map G_p and G_n to xbar
    if linear:
        weight_mapped = weight_map_linear_2source(weight_p=G_p, weight_n=G_n, bias_p=bias_p, bias_n=bias_n, xbar_row=xbar_row, xbar_column=xbar_column, Rm_max=Rm_max, Rm_off=Rm_off)
        
    else:
        weight_mapped = weight_map_conv_2source(weight_p=G_p, weight_n=G_n, bias_p=bias_p, bias_n=bias_n, xbar_row=xbar_row, xbar_column=xbar_column, Rm_max=Rm_max, Rm_off=Rm_off)
    
    return weight_mapped


# In[5]:


def matrix_A_generate(G_mapped,R_row,R_column):
    "generate matrix A for a group of xbars;G_mapped has shape of [i][j][r][c]"
    xbar_v_num = G_mapped.shape[0]
    xbar_h_num = G_mapped.shape[1]
    xbar_row = G_mapped.shape[2]
    xbar_column = G_mapped.shape[3]
    G = torch.zeros(xbar_v_num,xbar_h_num,xbar_row*xbar_column*2+xbar_row*2,xbar_row*xbar_column*2+xbar_row*2)
    for i in range(xbar_v_num):
        for j in range(xbar_h_num):
            G[i,j,:,:] = MNA_analysis(G_mapped[i,j],r_row=R_row,r_column=R_column)
    return G
            
def matrix_Ainv_generate_big(G_mapped,R_row,R_column):
    "generate matrix A for a group of xbars;G_mapped has shape of [i][j][r][c]"
    xbar_v_num = G_mapped.shape[0]
    xbar_h_num = G_mapped.shape[1]
    xbar_row = G_mapped.shape[2]
    xbar_column = G_mapped.shape[3]
    G = torch.zeros(xbar_v_num,xbar_h_num,xbar_row*xbar_column*2+xbar_row*2,xbar_row)
    for i in range(xbar_v_num):
        for j in range(xbar_h_num):
            G[i,j,:,:] = MNA_analysis_big(G_mapped[i,j],r_row=R_row,r_column=R_column)
    return G


def matrix_Ainv_generate_cuda(G_mapped,r_row,r_column):
    xbar_v_num = G_mapped.shape[0]
    xbar_h_num = G_mapped.shape[1] 
    xbar_row = G_mapped.shape[2]
    xbar_column = G_mapped.shape[3]
    
    A = torch.zeros(xbar_row*xbar_column*2+2*xbar_row,xbar_row*xbar_column*2+2*xbar_row)
    A_inv = torch.zeros(xbar_v_num,xbar_h_num,xbar_row*xbar_column*2+2*xbar_row,xbar_row).cuda()
    for i in range(xbar_v_num):
        for j in range(xbar_h_num):
            A = MNA_analysis_big(G_mapped[i,j,:,:],r_row,r_column)
            A = A.cuda()
            A_inv[i,j,:,:] = torch.linalg.inv(A)[:,-xbar_row:]
    return A_inv

def matrix_Ainv_generate_big_cuda(G_mapped,R_row,R_column):
    "generate matrix A for a group of xbars;G_mapped has shape of [i][j][r][c]"
    xbar_v_num = G_mapped.shape[0]
    xbar_h_num = G_mapped.shape[1]
    xbar_row = G_mapped.shape[2]
    xbar_column = G_mapped.shape[3]
    G = torch.zeros(xbar_v_num,xbar_h_num,xbar_row*xbar_column*2+xbar_row*2,xbar_row).cuda()
    for i in range(xbar_v_num):
        for j in range(xbar_h_num):
            G[i,j,:,:] = MNA_analysis_big_cuda(G_mapped[i,j],r_row=R_row,r_column=R_column)
            torch.cuda.empty_cache()
    return G

# In[6]:


def MNA_analysis(G_target,r_row,r_column):
    "generate A matrix for single xbar"
    xbar_row = G_target.shape[0]
    xbar_column = G_target.shape[1]
    
    G = torch.zeros((xbar_row*xbar_column*2+xbar_row,xbar_row*xbar_column*2+xbar_row))
   
    for i in range(xbar_row*xbar_column):
        #上节点第一列  
        if i%xbar_column==0:
            G[i][i] = 2/r_row + G_target[i//xbar_column][0]
            G[i][i+1] = -1/r_row
            G[i][i//xbar_column+xbar_row*xbar_column*2] = -1/r_row  ######此处可以考虑驱动的电阻
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][0]
        #上节点最后一列
        elif i%xbar_column==(xbar_column-1):
            G[i][i] = 1/r_row + G_target[i//xbar_column][xbar_column-1]
            G[i][i-1] = -1/r_row
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][xbar_column-1]
        #上节点非首尾列
        else:
            G[i][i] = 2/r_row + G_target[i//xbar_column][i%xbar_column]
            G[i][i-1] = -1/r_row
            G[i][i+1] = -1/r_row
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][i%xbar_column]

    ######下节点        
    for i in range(xbar_row*xbar_column,xbar_row*xbar_column*2):
        #下节点第一行
        if i<(xbar_row*xbar_column+xbar_column):
            G[i][i] = 1/r_column + G_target[0][i-xbar_row*xbar_column]
            G[i][i-xbar_row*xbar_column] = -G_target[0][i-xbar_row*xbar_column]
            G[i][i+xbar_column] = -1/r_column
        #下节点最后一行
        elif i>=(xbar_row*xbar_column*2-xbar_column):
            G[i][i] = 2/r_column + G_target[xbar_row-1][i+xbar_column-xbar_row*xbar_column*2] #######此处可考虑读出电路带来的负载电阻
            G[i][i-xbar_column] = -1/r_column
            G[i][i-xbar_row*xbar_column] = -G_target[xbar_row-1][i+xbar_column-xbar_row*xbar_column*2]
        #下节点非首尾行
        else:
            G[i][i] = 2/r_column + G_target[(i-xbar_row*xbar_column)//xbar_column][(i-xbar_row*xbar_column)%xbar_column]
            G[i][i-xbar_row*xbar_column] = -G_target[(i-xbar_row*xbar_column)//xbar_column][(i-xbar_row*xbar_column)%xbar_column]
            G[i][i-xbar_column] = -1/r_column
            G[i][i+xbar_column] = -1/r_column



    #####电源输入节点
    for i in range(xbar_row*xbar_column*2,xbar_row*xbar_column*2+xbar_row):
        G[i][i] = 1/r_row  ######此处可以考虑驱动的电阻
        G[i][(i-xbar_row*xbar_column*2)*xbar_column] = -1/r_row  ######可考虑驱动的电阻  
        
    B = torch.zeros((xbar_row*xbar_column*2+xbar_row,xbar_row))
    for i in range(xbar_row):
        B[i+xbar_row*xbar_column*2][i] = 1  #######单电源输入情况
    C = torch.transpose(B,0,1) 
    D = torch.zeros((xbar_row,xbar_row))
    temp1 = torch.hstack((G,B))
    temp2 = torch.hstack((C,D))
    A = torch.vstack((temp1,temp2))
    
    return A

#优化内存,直接生成A逆
def MNA_analysis_big(G_target,r_row,r_column):
    "generate A matrix for single xbar"
    xbar_row = G_target.shape[0]
    xbar_column = G_target.shape[1]
    
    G = torch.zeros((xbar_row*xbar_column*2+xbar_row,xbar_row*xbar_column*2+xbar_row))
   
    for i in range(xbar_row*xbar_column):
        #上节点第一列  
        if i%xbar_column==0:
            G[i][i] = 2/r_row + G_target[i//xbar_column][0]
            G[i][i+1] = -1/r_row
            G[i][i//xbar_column+xbar_row*xbar_column*2] = -1/r_row  ######此处可以考虑驱动的电阻
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][0]
        #上节点最后一列
        elif i%xbar_column==(xbar_column-1):
            G[i][i] = 1/r_row + G_target[i//xbar_column][xbar_column-1]
            G[i][i-1] = -1/r_row
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][xbar_column-1]
        #上节点非首尾列
        else:
            G[i][i] = 2/r_row + G_target[i//xbar_column][i%xbar_column]
            G[i][i-1] = -1/r_row
            G[i][i+1] = -1/r_row
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][i%xbar_column]

    ######下节点        
    for i in range(xbar_row*xbar_column,xbar_row*xbar_column*2):
        #下节点第一行
        if i<(xbar_row*xbar_column+xbar_column):
            G[i][i] = 1/r_column + G_target[0][i-xbar_row*xbar_column]
            G[i][i-xbar_row*xbar_column] = -G_target[0][i-xbar_row*xbar_column]
            G[i][i+xbar_column] = -1/r_column
        #下节点最后一行
        elif i>=(xbar_row*xbar_column*2-xbar_column):
            G[i][i] = 2/r_column + G_target[xbar_row-1][i+xbar_column-xbar_row*xbar_column*2] #######此处可考虑读出电路带来的负载电阻
            G[i][i-xbar_column] = -1/r_column
            G[i][i-xbar_row*xbar_column] = -G_target[xbar_row-1][i+xbar_column-xbar_row*xbar_column*2]
        #下节点非首尾行
        else:
            G[i][i] = 2/r_column + G_target[(i-xbar_row*xbar_column)//xbar_column][(i-xbar_row*xbar_column)%xbar_column]
            G[i][i-xbar_row*xbar_column] = -G_target[(i-xbar_row*xbar_column)//xbar_column][(i-xbar_row*xbar_column)%xbar_column]
            G[i][i-xbar_column] = -1/r_column
            G[i][i+xbar_column] = -1/r_column



    #####电源输入节点
    for i in range(xbar_row*xbar_column*2,xbar_row*xbar_column*2+xbar_row):
        G[i][i] = 1/r_row  ######此处可以考虑驱动的电阻
        G[i][(i-xbar_row*xbar_column*2)*xbar_column] = -1/r_row  ######可考虑驱动的电阻  
        
    B = torch.zeros((xbar_row*xbar_column*2+xbar_row,xbar_row))
    for i in range(xbar_row):
        B[i+xbar_row*xbar_column*2][i] = 1  #######单电源输入情况
    C = torch.transpose(B,0,1) 
    D = torch.zeros((xbar_row,xbar_row))
    temp1 = torch.hstack((G,B))
    temp2 = torch.hstack((C,D))
    A = torch.vstack((temp1,temp2))

    return A


#优化内存,直接生成A逆
def MNA_analysis_big_cuda(G_target,r_row,r_column):
    "generate A matrix for single xbar"
    r_row = torch.tensor(r_row).cuda()
    r_column = torch.tensor(r_column).cuda()
    xbar_row = G_target.shape[0]
    xbar_column = G_target.shape[1]
    
    G = torch.zeros((xbar_row*xbar_column*2+xbar_row,xbar_row*xbar_column*2+xbar_row)).cuda()
   
    for i in range(xbar_row*xbar_column):
        #上节点第一列  
        if i%xbar_column==0:
            G[i][i] = 2/r_row + G_target[i//xbar_column][0]
            G[i][i+1] = -1/r_row
            G[i][i//xbar_column+xbar_row*xbar_column*2] = -1/r_row  ######此处可以考虑驱动的电阻
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][0]
        #上节点最后一列
        elif i%xbar_column==(xbar_column-1):
            G[i][i] = 1/r_row + G_target[i//xbar_column][xbar_column-1]
            G[i][i-1] = -1/r_row
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][xbar_column-1]
        #上节点非首尾列
        else:
            G[i][i] = 2/r_row + G_target[i//xbar_column][i%xbar_column]
            G[i][i-1] = -1/r_row
            G[i][i+1] = -1/r_row
            G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][i%xbar_column]

    ######下节点        
    for i in range(xbar_row*xbar_column,xbar_row*xbar_column*2):
        #下节点第一行
        if i<(xbar_row*xbar_column+xbar_column):
            G[i][i] = 1/r_column + G_target[0][i-xbar_row*xbar_column]
            G[i][i-xbar_row*xbar_column] = -G_target[0][i-xbar_row*xbar_column]
            G[i][i+xbar_column] = -1/r_column
        #下节点最后一行
        elif i>=(xbar_row*xbar_column*2-xbar_column):
            G[i][i] = 2/r_column + G_target[xbar_row-1][i+xbar_column-xbar_row*xbar_column*2] #######此处可考虑读出电路带来的负载电阻
            G[i][i-xbar_column] = -1/r_column
            G[i][i-xbar_row*xbar_column] = -G_target[xbar_row-1][i+xbar_column-xbar_row*xbar_column*2]
        #下节点非首尾行
        else:
            G[i][i] = 2/r_column + G_target[(i-xbar_row*xbar_column)//xbar_column][(i-xbar_row*xbar_column)%xbar_column]
            G[i][i-xbar_row*xbar_column] = -G_target[(i-xbar_row*xbar_column)//xbar_column][(i-xbar_row*xbar_column)%xbar_column]
            G[i][i-xbar_column] = -1/r_column
            G[i][i+xbar_column] = -1/r_column



    #####电源输入节点
    for i in range(xbar_row*xbar_column*2,xbar_row*xbar_column*2+xbar_row):
        G[i][i] = 1/r_row  ######此处可以考虑驱动的电阻
        G[i][(i-xbar_row*xbar_column*2)*xbar_column] = -1/r_row  ######可考虑驱动的电阻  
        
    B = torch.zeros((xbar_row*xbar_column*2+xbar_row,xbar_row)).cuda()
    for i in range(xbar_row):
        B[i+xbar_row*xbar_column*2][i] = 1  #######单电源输入情况
    C = torch.transpose(B,0,1) 
    D = torch.zeros((xbar_row,xbar_row)).cuda()
    temp1 = torch.hstack((G,B))
    temp2 = torch.hstack((C,D))
    del G,B,C,D
    torch.cuda.empty_cache()
    G = torch.vstack((temp1,temp2))
    del temp1,temp2
    A_inv = torch.linalg.inv(G)
    del G
    torch.cuda.empty_cache()
    A_inv_reduced = A_inv[:,-xbar_row:].clone().detach()
    del A_inv
    torch.cuda.empty_cache()
    return A_inv_reduced

#快速生成A逆，占用大内存
# def MNA_analysis_big_cuda(G_target,r_row,r_column):
#     "generate A matrix for single xbar"
#     r_row = torch.tensor(r_row).cuda()
#     r_column = torch.tensor(r_column).cuda()
#     xbar_row = G_target.shape[0]
#     xbar_column = G_target.shape[1]
    
#     G = torch.zeros((xbar_row*xbar_column*2+xbar_row,xbar_row*xbar_column*2+xbar_row)).cuda()
   
#     for i in range(xbar_row*xbar_column):
#         #上节点第一列  
#         if i%xbar_column==0:
#             G[i][i] = 2/r_row + G_target[i//xbar_column][0]
#             G[i][i+1] = -1/r_row
#             G[i][i//xbar_column+xbar_row*xbar_column*2] = -1/r_row  ######此处可以考虑驱动的电阻
#             G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][0]
#         #上节点最后一列
#         elif i%xbar_column==(xbar_column-1):
#             G[i][i] = 1/r_row + G_target[i//xbar_column][xbar_column-1]
#             G[i][i-1] = -1/r_row
#             G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][xbar_column-1]
#         #上节点非首尾列
#         else:
#             G[i][i] = 2/r_row + G_target[i//xbar_column][i%xbar_column]
#             G[i][i-1] = -1/r_row
#             G[i][i+1] = -1/r_row
#             G[i][i+xbar_row*xbar_column] = -G_target[i//xbar_column][i%xbar_column]

#     ######下节点        
#     for i in range(xbar_row*xbar_column,xbar_row*xbar_column*2):
#         #下节点第一行
#         if i<(xbar_row*xbar_column+xbar_column):
#             G[i][i] = 1/r_column + G_target[0][i-xbar_row*xbar_column]
#             G[i][i-xbar_row*xbar_column] = -G_target[0][i-xbar_row*xbar_column]
#             G[i][i+xbar_column] = -1/r_column
#         #下节点最后一行
#         elif i>=(xbar_row*xbar_column*2-xbar_column):
#             G[i][i] = 2/r_column + G_target[xbar_row-1][i+xbar_column-xbar_row*xbar_column*2] #######此处可考虑读出电路带来的负载电阻
#             G[i][i-xbar_column] = -1/r_column
#             G[i][i-xbar_row*xbar_column] = -G_target[xbar_row-1][i+xbar_column-xbar_row*xbar_column*2]
#         #下节点非首尾行
#         else:
#             G[i][i] = 2/r_column + G_target[(i-xbar_row*xbar_column)//xbar_column][(i-xbar_row*xbar_column)%xbar_column]
#             G[i][i-xbar_row*xbar_column] = -G_target[(i-xbar_row*xbar_column)//xbar_column][(i-xbar_row*xbar_column)%xbar_column]
#             G[i][i-xbar_column] = -1/r_column
#             G[i][i+xbar_column] = -1/r_column



#     #####电源输入节点
#     for i in range(xbar_row*xbar_column*2,xbar_row*xbar_column*2+xbar_row):
#         G[i][i] = 1/r_row  ######此处可以考虑驱动的电阻
#         G[i][(i-xbar_row*xbar_column*2)*xbar_column] = -1/r_row  ######可考虑驱动的电阻  
        
#     B = torch.zeros((xbar_row*xbar_column*2+xbar_row,xbar_row)).cuda()
#     for i in range(xbar_row):
#         B[i+xbar_row*xbar_column*2][i] = 1  #######单电源输入情况
#     C = torch.transpose(B,0,1) 
#     D = torch.zeros((xbar_row,xbar_row)).cuda()
#     temp1 = torch.hstack((G,B))
#     temp2 = torch.hstack((C,D))
#     A_inv = torch.linalg.inv(torch.vstack((temp1,temp2)))
#     A_inv_reduced = A_inv[:,-xbar_row:].clone().detach()
#     del A_inv
#     gc.collect()
#     torch.cuda.empty_cache()
#     return A_inv_reduced



# # 对每层网络进行映射并生成MNA算法中A矩阵
# **对于模型参数加载torch.load(PATH),每层参数分别以weight和bias保存为tensor形式**<br>
# **1.将model.state_dict()中的weight,bias参数传给函数W_to_G(weight,bias,Rm_min,Rm_max,Rm_off,xbar_row,xbar_column,linear),将weight和bias分为正负两组数据，并转换为电导矩阵G**<br>
# **2.函数W_to_G在内部调用weight_map_conv/linear函数，将G_p/bias_p和G_n/bias_n整合映射，并按常规方法映射到xbar上,返回weight_mapped_p和weight_mapped_n两组xbars(shape为i,j,r,c)**<br>
# **3.使用matrix_A_generate(G_mapped,R_row,R_column)函数根据传入的映射完成的xbars(G_mapped)，按照MNA算法生成A 矩阵，其中每个子xbar内部调用MNA_analysis函数生成A矩阵，组成4维A矩阵**<br>

# In[8]:


# 例子展示
# weight = model_param['conv1.weight']
# bias = model_param['conv1.bias']

# G_p,G_n = W_to_G(weight,bias,linear=False)
# print(G_p.shape,G_n.shape)
# A_p = matrix_A_generate(G_p)
# A_n = matrix_A_generate(G_n)
# # print(weight,bias)
# print(A_p.shape)


# In[97]:


#vector_z_generate_linear函数处理线性层的一个输入特征向量，将其划分为若干个A逆矩阵的输入
def vector_z_generate_linear(feature_map,xbar_row,xbar_column,bias=True):
    "generate vector z according to feature map;vector z in shape of torch.hstack((torch.zeros(xbar_row*xbar_column*2+xbar_row),(tensor.size(xbar_row)))"
    if bias:
        feature_map = torch.hstack((feature_map,torch.ones(1)))
        
    xbar_v_num = (feature_map.shape[-1]-1)//xbar_row+1
    feature_map = torch.hstack((feature_map,torch.zeros(xbar_v_num*xbar_row-feature_map.shape[-1])))
    z_set = torch.zeros(xbar_v_num, xbar_row*xbar_column*2+2*xbar_row)
    for i in range(xbar_v_num):
        z_set[i,xbar_row*xbar_column*2+xbar_row:] = feature_map[i*xbar_row:(i+1)*xbar_row]
    return z_set

def vector_z_generate_conv(feature_map,xbar_row,xbar_column,kernel_h,kernel_w,stride_h=1,stride_w=1,padding = 0,bias = True):
    "generate input vector set for conv layer"
    #finish FM_in padding and FM_out size esitimation
    pad = torch.nn.ZeroPad2d(padding)
    feature_map = pad(feature_map)
    xbar_v_num = (feature_map.shape[0]*kernel_h*kernel_w+bias-1)//xbar_row+1
    FM_out_h = int((feature_map.shape[-2]-kernel_h)/stride_h+1)
    FM_out_w = int((feature_map.shape[-1]-kernel_w)/stride_w+1)
    z_set = torch.zeros(FM_out_h,FM_out_w,xbar_v_num,xbar_row*xbar_column*2+2*xbar_row)
    for i in range(FM_out_h):
        for j in range(FM_out_w):
            feature_map_slice = feature_map[:,i*stride_h:i*stride_h+kernel_h,j*stride_w:j*stride_w+kernel_w].reshape(-1)
            z_set[i,j,:,:] = vector_z_generate_linear(feature_map_slice,xbar_row=xbar_row,xbar_column=xbar_column,bias = bias)
            
    return z_set


def vector_z_generate_linear_quick(feature_map,xbar_row,xbar_column,bias=True):
    "generate vector z according to feature map;vector z in shape of torch.hstack((torch.zeros(xbar_row*xbar_column*2+xbar_row),(tensor.size(xbar_row)))"
    if bias:
        feature_map = torch.hstack((feature_map,torch.ones(1)))
        
    xbar_v_num = (feature_map.shape[-1]-1)//xbar_row+1
    z_set = torch.hstack((feature_map,torch.zeros(xbar_v_num*xbar_row-feature_map.shape[-1])))

    return z_set

def vector_z_generate_conv_quick(feature_map,xbar_row,xbar_column,kernel_h,kernel_w,stride_h=1,stride_w=1,padding = 0,bias = True):
    "generate input vector set for conv layer"
    #finish FM_in padding and FM_out size esitimation
    pad = torch.nn.ZeroPad2d(padding)
    feature_map = pad(feature_map)
    xbar_v_num = (feature_map.shape[0]*kernel_h*kernel_w+bias-1)//xbar_row+1
    FM_out_h = int((feature_map.shape[-2]-kernel_h)/stride_h+1)
    FM_out_w = int((feature_map.shape[-1]-kernel_w)/stride_w+1)
    z_set = torch.zeros(FM_out_h,FM_out_w,xbar_v_num*xbar_row)
    for i in range(FM_out_h):
        for j in range(FM_out_w):
            feature_map_slice = feature_map[:,i*stride_h:i*stride_h+kernel_h,j*stride_w:j*stride_w+kernel_w].reshape(-1)
            z_set[i,j,:] = vector_z_generate_linear_quick(feature_map_slice,xbar_row=xbar_row,xbar_column=xbar_column,bias = bias)
            
    return z_set
    
    
def vector_z_generate_linear_cuda(feature_map,xbar_row,xbar_column,bias=True):
    "generate vector z according to feature map;vector z in shape of torch.hstack((torch.zeros(xbar_row*xbar_column*2+xbar_row),(tensor.size(xbar_row)))"
    if bias:
        feature_map = torch.hstack((feature_map,torch.ones(1).cuda()))
        
    xbar_v_num = (feature_map.shape[-1]-1)//xbar_row+1
    z_set = torch.hstack((feature_map,torch.zeros(xbar_v_num*xbar_row-feature_map.shape[-1]).cuda()))

    return z_set
    
def vector_z_generate_conv_cuda(feature_map,xbar_row,xbar_column,kernel_h,kernel_w,stride_h=1,stride_w=1,padding = 0,bias = True):
    "generate input vector set for conv layer"
    #finish FM_in padding and FM_out size esitimation
    pad = torch.nn.ZeroPad2d(padding)
    feature_map = pad(feature_map)
    xbar_v_num = (feature_map.shape[0]*kernel_h*kernel_w+bias-1)//xbar_row+1
    FM_out_h = int((feature_map.shape[-2]-kernel_h)/stride_h+1)
    FM_out_w = int((feature_map.shape[-1]-kernel_w)/stride_w+1)
    z_set = torch.zeros(FM_out_h,FM_out_w,xbar_v_num*xbar_row).cuda()
    for i in range(FM_out_h):
        for j in range(FM_out_w):
            feature_map_slice = feature_map[:,i*stride_h:i*stride_h+kernel_h,j*stride_w:j*stride_w+kernel_w].reshape(-1)
            z_set[i,j,:] = vector_z_generate_linear_cuda(feature_map_slice,xbar_row=xbar_row,xbar_column=xbar_column,bias = bias)
            
    return z_set    


def vector_z_generate_linear_cuda_2source(feature_map,xbar_row,xbar_column,bias=True):
    "generate vector z according to feature map;vector z in shape of torch.hstack((torch.zeros(xbar_row*xbar_column*2+xbar_row),(tensor.size(xbar_row)))"
    if bias:
        feature_map = torch.hstack((feature_map,torch.ones(1).cuda()))
        
    xbar_v_num = (feature_map.shape[-1]*2-1)//xbar_row+1
    feature_map = torch.stack((feature_map,-feature_map),-1).reshape(-1)
    z_set = torch.hstack((feature_map,torch.zeros(xbar_v_num*xbar_row-feature_map.shape[-1]).cuda()))

    return z_set

def vector_z_generate_conv_cuda_2source(feature_map,xbar_row,xbar_column,kernel_h,kernel_w,stride_h=1,stride_w=1,padding = 0,bias = True):
    "generate input vector set for conv layer"
    #finish FM_in padding and FM_out size esitimation
    pad = torch.nn.ZeroPad2d(padding)
    feature_map = pad(feature_map)
    xbar_v_num = ((feature_map.shape[0]*kernel_h*kernel_w+bias)*2-1)//xbar_row+1
    FM_out_h = int((feature_map.shape[-2]-kernel_h)/stride_h+1)
    FM_out_w = int((feature_map.shape[-1]-kernel_w)/stride_w+1)
    z_set = torch.zeros(FM_out_h,FM_out_w,xbar_v_num*xbar_row).cuda()
    for i in range(FM_out_h):
        for j in range(FM_out_w):
            feature_map_slice = feature_map[:,i*stride_h:i*stride_h+kernel_h,j*stride_w:j*stride_w+kernel_w].reshape(-1)
            z_set[i,j,:] = vector_z_generate_linear_cuda_2source(feature_map_slice,xbar_row=xbar_row,xbar_column=xbar_column,bias = bias)
            
    return z_set  

# # 根据输入feature map生成mna算法中对应的z矩阵
# **vector_z_generate_linear(feature_map,xbar_row,xbar_column,bias=True)** <br> **输入的feature map为行向量，根据阵列大小和映射关系，将输入特征图划分为对应vin[xbar_v_num][xbar_row]，再转换为对应的z_set[xbar_v_num][xbar_row\*xbar_column \*2+2\*xbar_column]**<br>
# <br><br>
# **vector_z_generate_conv(feature_map,xbar_row,xbar_column,kernel_h,kernel_w,stride_h=1,stride_w=1,padding = 0,bias = True)**<br>
# **输入的特征图形如[Cin][height][width],函数先对特征图padding，再根据核的大小与stride大小生成每次卷积所需的特征图slice，** <br> 
# **转为行向量，内部调用函数vector_z_generate_linear()生成对应的z_set[FM_out_h][FM_out_w][xbar_v_num][xbar_row\*xbar_column\*2+2\*xbar_column]**

# In[10]:


# FM = torch.arange(6*12*12).reshape(6,12,12)
# set = vector_z_generate_conv(FM,xbar_row=6,xbar_column=6,kernel_h=5,kernel_w=5)
# print(set.shape)
# print(set[1,0,:,78:])


# In[101]:
#################################################################################################################################
#for cpu
def conv_forward(A_inv,z_set,G_mapped,Cout,xbar_row,xbar_column,bias=True):
    Iout = torch.zeros(Cout*z_set.shape[0]*z_set.shape[1])
    
    for i in range(z_set.shape[0]):
        for j in range(z_set.shape[1]):
            Iout[(i*z_set.shape[1]+j)*Cout:(i*z_set.shape[1]+j+1)*Cout] = linear_forward(A_inv=A_inv,z_set=z_set[i,j,:,:],G_mapped=G_mapped,out_neuron_num=Cout,xbar_row=xbar_row,xbar_column=xbar_column,bias=bias)

    Iout_reorder = torch.zeros(Cout,z_set.shape[0],z_set.shape[1])
    
    for k in range(Cout):
        Iout_reorder[k,:,:] = Iout[k::Cout].reshape(z_set.shape[0],z_set.shape[1])

    return Iout_reorder


def linear_forward(A_inv,z_set,G_mapped,out_neuron_num,xbar_row,xbar_column,bias=True):
#     z_set = vector_z_generate_linear(feature_map=feature_map,xbar_row=xbar_row,xbar_column=xbar_column,bias=bias)
    Iout_sum = torch.zeros(A_inv.shape[0],A_inv.shape[1]*xbar_column)
    for i in range(A_inv.shape[0]):
        for j in range(A_inv.shape[1]):
            x = A_inv[i,j,:,:]@z_set[i,:]
            Vrram = x[:xbar_row*xbar_column].reshape(xbar_row,xbar_column)-x[xbar_row*xbar_column:xbar_row*xbar_column*2].reshape(xbar_row,xbar_column)
            Iout_sum[i,j*xbar_column:(j+1)*xbar_column] = (Vrram*G_mapped[i,j,:,:]).sum(axis=0)
    Iout = Iout_sum.sum(axis=0)[:out_neuron_num]
    
    return Iout





def conv_forward_quick(A_inv,z_set,G_mapped,Cout,xbar_row,xbar_column,bias=True):
    Iout = torch.zeros(Cout*z_set.shape[0]*z_set.shape[1])
    
    for i in range(z_set.shape[0]):
        for j in range(z_set.shape[1]):
            Iout[(i*z_set.shape[1]+j)*Cout:(i*z_set.shape[1]+j+1)*Cout] = linear_forward_quick(A_inv=A_inv,z_set=z_set[i,j,:],G_mapped=G_mapped,out_neuron_num=Cout,xbar_row=xbar_row,xbar_column=xbar_column,bias=bias)
    Iout_reorder = torch.zeros(Cout,z_set.shape[0],z_set.shape[1])
    
    for k in range(Cout):
        Iout_reorder[k,:,:] = Iout[k::Cout].reshape(z_set.shape[0],z_set.shape[1])

    return Iout_reorder


def linear_forward_quick(A_inv,z_set,G_mapped,out_neuron_num,xbar_row,xbar_column,bias=True):
#     z_set = vector_z_generate_linear(feature_map=feature_map,xbar_row=xbar_row,xbar_column=xbar_column,bias=bias)
    Iout_sum = torch.zeros(A_inv.shape[0],A_inv.shape[1]*xbar_column)
    for i in range(A_inv.shape[0]):
        for j in range(A_inv.shape[1]):
            x = A_inv[i,j,:,-xbar_row:]@z_set[i*xbar_row:(i+1)*xbar_row]
            Vrram = x[:xbar_row*xbar_column].reshape(xbar_row,xbar_column)-x[xbar_row*xbar_column:xbar_row*xbar_column*2].reshape(xbar_row,xbar_column)
            Iout_sum[i,j*xbar_column:(j+1)*xbar_column] = (Vrram*G_mapped[i,j,:,:]).sum(axis=0)
    Iout = Iout_sum.sum(axis=0)[:out_neuron_num]
    
    return Iout

############################################################################################################################
# for GPU
# def linear_forward_cuda(A_inv,z_set,G_mapped,out_neuron_num,xbar_row,xbar_column,bias=True):
# #     z_set = vector_z_generate_linear(feature_map=feature_map,xbar_row=xbar_row,xbar_column=xbar_column,bias=bias)
#     Iout_sum = torch.zeros(A_inv.shape[0],A_inv.shape[1]*xbar_column).cuda()
#     for i in range(A_inv.shape[0]):
#         for j in range(A_inv.shape[1]):
#             x = A_inv[i,j,:,-xbar_row:]@z_set[i*xbar_row:(i+1)*xbar_row]
#             Vrram = x[:xbar_row*xbar_column].reshape(xbar_row,xbar_column)-x[xbar_row*xbar_column:xbar_row*xbar_column*2].reshape(xbar_row,xbar_column)
#             Iout_sum[i,j*xbar_column:(j+1)*xbar_column] = (Vrram*G_mapped[i,j,:,:]).sum(axis=0)
#     Iout = Iout_sum.sum(axis=0)[:out_neuron_num]
    
#     return Iout

# def conv_forward_cuda(A_inv,z_set,G_mapped,Cout,xbar_row,xbar_column,bias=True):
#     Iout = torch.zeros(Cout*z_set.shape[0]*z_set.shape[1]).cuda()
    
#     for i in range(z_set.shape[0]):
#         for j in range(z_set.shape[1]):
#             Iout[(i*z_set.shape[1]+j)*Cout:(i*z_set.shape[1]+j+1)*Cout] = linear_forward_cuda(A_inv=A_inv,z_set=z_set[i,j,:],G_mapped=G_mapped,out_neuron_num=Cout,xbar_row=xbar_row,xbar_column=xbar_column,bias=bias)
#     Iout_reorder = torch.zeros(Cout,z_set.shape[0],z_set.shape[1]).cuda()
    
#     for k in range(Cout):
#         Iout_reorder[k,:,:] = Iout[k::Cout].reshape(z_set.shape[0],z_set.shape[1])

#     return Iout_reorder

def linear_forward_cuda_quick(A_inv,z_set,G_mapped,out_neuron_num,xbar_row,xbar_column,bias=True):
#     z_set = vector_z_generate_linear(feature_map=feature_map,xbar_row=xbar_row,xbar_column=xbar_column,bias=bias)
    Iout_sum = torch.zeros(A_inv.shape[0],A_inv.shape[1]*xbar_column).cuda()
    for i in range(A_inv.shape[0]):
        x = A_inv[i,:,:,-xbar_row:]@z_set[i*xbar_row:(i+1)*xbar_row]
        Vrram = (x[:,:xbar_row*xbar_column]-x[:,xbar_row*xbar_column:xbar_row*xbar_column*2]).reshape(A_inv.shape[1],xbar_row,xbar_column)
        Iout_sum[i,:] = (Vrram*G_mapped[i,:,:,:]).sum(axis=1).reshape(-1)
    Iout = Iout_sum.sum(axis=0)[:out_neuron_num]
    
    return Iout

def conv_forward_cuda_quick(A_inv,z_set,G_mapped,Cout,xbar_row,xbar_column,bias=True):
    Iout = torch.zeros(Cout*z_set.shape[0]*z_set.shape[1]).cuda()
    
    for i in range(z_set.shape[0]):
        for j in range(z_set.shape[1]):
            Iout[(i*z_set.shape[1]+j)*Cout:(i*z_set.shape[1]+j+1)*Cout] = linear_forward_cuda_quick(A_inv=A_inv,z_set=z_set[i,j,:],G_mapped=G_mapped,out_neuron_num=Cout,xbar_row=xbar_row,xbar_column=xbar_column,bias=bias)
    Iout_reorder = torch.zeros(Cout,z_set.shape[0],z_set.shape[1]).cuda()
    
    for k in range(Cout):
        Iout_reorder[k,:,:] = Iout[k::Cout].reshape(z_set.shape[0],z_set.shape[1])

    return Iout_reorder


# def linear_forward_cuda(A_inv,z_set,G_mapped,out_neuron_num):
#     "z_set (xbar_v_num*xbar_row),to z_set (xbar_v_num,xbar_h_num,xbar_row,1)"
#     xbar_v_num, xbar_h_num = G_mapped.shape[0], G_mapped.shape[1]
#     xbar_row, xbar_column = G_mapped.shape[2], G_mapped.shape[3]
#     z_set = z_set.reshape(xbar_v_num,xbar_row).unsqueeze(1).repeat(1,xbar_h_num,1).unsqueeze(3)
#     x = A_inv@z_set
#     Iout = ((x[:,:,:xbar_row*xbar_column,:]-x[:,:,xbar_row*xbar_column:xbar_row*xbar_column*2,:]).reshape(xbar_v_num,xbar_h_num,xbar_row,xbar_column)*G_mapped).sum(0).sum(1).reshape(-1)
#     return Iout[:out_neuron_num]

# def conv_forward_cuda(A_inv,z_set,G_mapped,Cout):
#     fm_out_h, fm_out_w = z_set.shape[0], z_set.shape[1]
#     Iout = torch.zeros(Cout,fm_out_h,fm_out_w).cuda()
#     for i in range(fm_out_h):
#         for j in range(fm_out_w):
#             Iout[:,i,j] = linear_forward_cuda(A_inv,z_set[i,j,:],G_mapped,out_neuron_num=Cout)
#     return Iout


def linear_forward_cuda(A_inv,z_set,G_mapped_shape,r_column,out_neuron_num):
    "z_set (xbar_v_num*xbar_row),to z_set (xbar_v_num,xbar_h_num,xbar_row,1)"
    xbar_v_num, xbar_h_num = G_mapped_shape[0], G_mapped_shape[1]
    xbar_row, xbar_column = G_mapped_shape[2], G_mapped_shape[3]
    z_set = z_set.reshape(xbar_v_num,xbar_row).unsqueeze(1).repeat(1,xbar_h_num,1).unsqueeze(3)
    x = A_inv@z_set
    Iout = ((x[:,:,xbar_row*xbar_column*2-xbar_column:xbar_row*xbar_column*2,:]/r_column)).sum(0).reshape(-1)
    return Iout[:out_neuron_num]

def conv_forward_cuda(A_inv,z_set,G_mapped_shape,r_column,Cout):
    fm_out_h, fm_out_w = z_set.shape[0], z_set.shape[1]
    Iout = torch.zeros(Cout,fm_out_h,fm_out_w).cuda()
    for i in range(fm_out_h):
        for j in range(fm_out_w):
            Iout[:,i,j] = linear_forward_cuda(A_inv,z_set[i,j,:],G_mapped_shape,r_column,out_neuron_num=Cout)
    return Iout


def linear_forward_cuda_2source(A_inv,z_set,R_column,out_neuron_num):
    "z_set (xbar_v_num*xbar_row),to z_set (xbar_v_num,xbar_h_num,xbar_row,1)"
    xbar_v_num, xbar_h_num = A_inv.shape[0], A_inv.shape[1]
    xbar_row, xbar_column = A_inv.shape[3], int((A_inv.shape[2]-2*A_inv.shape[3])/A_inv.shape[3]/2)
    z_set = z_set.reshape(xbar_v_num,xbar_row).unsqueeze(1).repeat(1,xbar_h_num,1).unsqueeze(3)
    x = A_inv@z_set
    Iout = ((x[:,:,xbar_row*xbar_column*2-xbar_column:xbar_row*xbar_column*2,:]).reshape(xbar_v_num,xbar_h_num,xbar_column)/R_column).sum(0).reshape(-1)
    return Iout[:out_neuron_num]

def conv_forward_cuda_2source(A_inv,z_set,R_column,Cout):
    fm_out_h, fm_out_w = z_set.shape[0], z_set.shape[1]
    Iout = torch.zeros(Cout,fm_out_h,fm_out_w).cuda()
    for i in range(fm_out_h):
        for j in range(fm_out_w):
            Iout[:,i,j] = linear_forward_cuda_2source(A_inv,z_set[i,j,:],R_column=R_column,out_neuron_num=Cout)
    return Iout

# # 需要内存太大(fm_out_h,fm_out_w,xbar_v_num,xbar_h_num,8320,xbar_row)
# def conv_forward_cuda(A_inv,z_set,G_mapped,Cout):
#     "z_set (fm_out_h,fm_out_w,xbar_v_num*xbar_row),to z_set (fm_out_h,fm_out_w,xbar_v_num,xbar_h_num,xbar_row,1)"
#     fm_out_h, fm_out_w = z_set.shape[0], z_set.shape[1]
#     xbar_v_num, xbar_h_num = G_mapped.shape[0], G_mapped.shape[1]
#     xbar_row, xbar_column = G_mapped.shape[2], G_mapped.shape[3]
#     z_set = z_set.reshape(fm_out_h,fm_out_w,xbar_v_num,xbar_row).unsqueeze(3).repeat(1,1,1,xbar_h_num,1).unsqueeze(5)
#     x = A_inv@z_set
#     Iout = (((x[:,:,:,:,:xbar_row*xbar_column,:]-x[:,:,:,:,xbar_row*xbar_column:xbar_row*xbar_column*2,:]).reshape(fm_out_h,fm_out_w,xbar_v_num,xbar_h_num,xbar_row,xbar_column)*G_mapped).sum(4).sum(2).reshape(fm_out_h,fm_out_w,-1))[:,:,:Cout]
#     Iout = Iout.permute(2,0,1)
#     return Iout



def conv_forward_cuda1(A_inv,z_set,G_mapped,Cout):
    "z_set (fm_out_h,fm_out_w,xbar_v_num*xbar_row),to z_set (fm_out_h,fm_out_w,xbar_v_num,xbar_h_num,xbar_row,1)"
    fm_out_h, fm_out_w = z_set.shape[0], z_set.shape[1]
    xbar_v_num, xbar_h_num = G_mapped.shape[0], G_mapped.shape[1]
    xbar_row, xbar_column = G_mapped.shape[2], G_mapped.shape[3]
    z_set = z_set.reshape(fm_out_h,fm_out_w,xbar_v_num,xbar_row).unsqueeze(3).repeat(1,1,1,xbar_h_num,1).unsqueeze(5)
    Iout = torch.zeros(fm_out_h,fm_out_w,Cout).cuda()
    for i in range(fm_out_h):     
        x = A_inv@z_set[i]
        Iout[i,:,:] = (((x[:,:,:,:xbar_row*xbar_column,:]-x[:,:,:,xbar_row*xbar_column:xbar_row*xbar_column*2,:]).reshape(fm_out_w,xbar_v_num,xbar_h_num,xbar_row,xbar_column)*G_mapped).sum(3).sum(1).reshape(fm_out_w,-1))[:,:Cout]
    Iout = Iout.permute(2,0,1)
    return Iout


# # 卷积与线性层前向过程
# **linear_forward()函数输入 A_inv[xbar_v_num][xbar_h_num][xbar_row\*xbar_column\*2+xbar_row\*2][xbar_row\*xbar_column\*2+xbar_row\*2]  z_set[xbar_v_num][xbar_row\*xbar_column\*2+xbar_row\*2]**<br><br>
# **conv_forward()函数输入 A_inv[xbar_v_num][xbar_h_num][xbar_row\*xbar_column\*2+xbar_row\*2][xbar_row\*xbar_column\*2+xbar_row\*2] z_set[FM_out_h][FM_out_w][xbar_v_num][xbar_row\*xbar_column\*2+xbar_row\*2]**<br> 

# In[102]:


# weight = torch.rand(16,6,5,5)
# bias = torch.rand(16)
# Vin = torch.rand(6,14,14)
# G_p,G_n = W_to_G(weight,bias,Rm_min=10e3,Rm_max=50e3,Rm_off=10e9,xbar_row=64,xbar_column=64,linear=False)
# print(G_p.shape,G_p.shape)
# A_p = matrix_A_generate(G_p)
# A_n = matrix_A_generate(G_n)
# print(A_p.shape,A_n.shape)
# A_p_inv = torch.linalg.inv(A_p)
# A_n_inv = torch.linalg.inv(A_n)

# #验证原始卷积推理用时
# z_set = vector_z_generate_conv(Vin,xbar_row=64,xbar_column=64,kernel_h=5,kernel_w=5,stride_h=1,stride_w=1,padding = 0,bias = True)
# start_time1 = time.time()
# I_outp = conv_forward(A_p_inv,z_set,G_p,Cout=16,xbar_row=64,xbar_column=64,bias=True)
# end_time1 = time.time()
# print(end_time1-start_time1)

# #验证快速卷积推理用时
# z_set_q = vector_z_generate_conv_quick(Vin,xbar_row=64,xbar_column=64,kernel_h=5,kernel_w=5,stride_h=1,stride_w=1,padding = 0,bias = True)
# start_time2 = time.time()
# I_outp_q = conv_forward_quick(A_p_inv,z_set_q,G_p,Cout=16,xbar_row=64,xbar_column=64,bias=True)
# end_time2 = time.time()
# print(end_time2-start_time2)
# print((I_outp==I_outp_q).sum().sum().sum())


# In[107]:


# weight = torch.rand(120,400)
# bias = torch.rand(120)
# Vin = torch.rand(400)
# G_p,G_n = W_to_G(weight,bias,Rm_min=10e3,Rm_max=50e3,Rm_off=10e9,xbar_row=64,xbar_column=64,linear=False)
# print(G_p.shape,G_p.shape)
# A_p = matrix_A_generate(G_p)
# A_n = matrix_A_generate(G_n)
# print(A_p.shape,A_n.shape)
# A_p_inv = torch.linalg.inv(A_p)
# A_n_inv = torch.linalg.inv(A_n)

# #验证原始linear推理用时
# z_set = vector_z_generate_linear(Vin,xbar_row=64,xbar_column=64,bias = True)
# start_time1 = time.time()
# I_outp = linear_forward(A_p_inv,z_set,G_p,out_neuron_num=120,xbar_row=64,xbar_column=64,bias=True)
# end_time1 = time.time()
# print(end_time1-start_time1)

# #验证快速linear推理用时
# z_set_q = vector_z_generate_linear_quick(Vin,xbar_row=64,xbar_column=64,bias = True)
# start_time2 = time.time()
# I_outp_q = linear_forward_quick(A_p_inv,z_set_q,G_p,out_neuron_num=120,xbar_row=64,xbar_column=64,bias=True)
# end_time2 = time.time()
# print(end_time2-start_time2)
# print(I_outp==I_outp_q)


# In[14]:


# Rm_min, Rm_max, Rm_off = 10e3, 50e3, 10e9
# xbar_row, xbar_column = 64,64
# R_row, R_column = 1,1
# show_memory_info('initial')

# w = torch.rand(64*64*3*3).reshape(64,64,3,3)
# bias = torch.rand(64)
# G_p, G_n = W_to_G(w, bias, Rm_min, Rm_max, Rm_off, xbar_row, xbar_column, linear=False)
# show_memory_info('step1')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




