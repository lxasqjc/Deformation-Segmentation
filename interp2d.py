
import warnings

import math
from torch import nn
from torch.autograd import Function
import torch
import spatial.qhull as qhull
import numpy as np
import torch.nn.functional as F



class Interp2D(nn.Module):
    '''
    New 2d Interpolation in Pytorch
    Reference to scipy.griddata
    Args；
        h, w:  height,width of input
        points: points to interpote shape: [num, 2]
        values:  values of points shape:[num, valuedim]
    return:
       2D interpolate result, shape: [valuedim, h, w]
    '''
    def __init__(self, h, w, add_corner=False):
        super(Interp2D,self).__init__()
        row_coord = np.arange(h).repeat([w]).reshape([h,w])
        col_coord = np.arange(w).repeat([h]).reshape([w,h]).T
        self.coord = np.stack([row_coord, col_coord])
        self.coord = self.coord.transpose([1,2,0]).reshape([-1,2])
        self.add_corner = add_corner
        self.w = w
        self.h = h
        # if self.add_corner==False:
        #     raise Exception('Now add_corner must be true')

    def forward(self, points, values):
        '''
        notes for gradients: numpy based qhull operations find traingular
        simplices (tri_map --- corner locations) and weights for interpolation,
        tri_map and weights are not derivable, but it's ok, because qhull
        traingular operation is deterministic and we don't need to learn
        parameters for it.

        While gradients still flow because we never put values to cpu, we only
        use tri_map to sample pixels from values, which always on gpu.
        '''
        if self.add_corner:
            points = torch.cat([points.cpu(), torch.Tensor([[0,0], [0, self.w-1],
                                  [self.h-1,0], [self.h-1, self.w-1]]).long()], dim=0)
            values = torch.cat([values, torch.zeros([4,values.shape[1]]).to(values.device)], dim=0)
        else:
            points = points.cpu()
           # Add 4 zeros corner points
        self.tri = qhull.Delaunay(points)
        vdim = values.shape[-1]
        # print('points_shape: {}'.format(points.shape))
        isimplex, weights = self.tri.find_simplex(self.coord, return_c=True)
        # attempt to correct CUDA error: device-side assert triggered
        # which may caused by Points outside the triangulation get the value -1.
        if np.sum(isimplex==-1)>0:
            print('WARNING: {} Points outside the triangulation get the value -1, multiplied by 0\n'.format(np.sum(isimplex==-1)))
            isimplex[isimplex==-1] *= 0
        #the array `weights` is filled with the corresponding barycentric coordinates.
        weights = torch.from_numpy(weights).float().to(values.device)
        # print('isimplex_shape original: {}'.format(isimplex.shape))
        isimplex = torch.from_numpy(isimplex).to(values.device)
        isimplex = isimplex.long()
        isimplex = isimplex.reshape([-1,1])
        # print('isimplex_shape: {}, weights_shape: {}'.format(isimplex.shape, weights.shape))

        # shape: isimplex: [h*w,1]      c: [h,w,c]

        simplices =torch.from_numpy(self.tri.simplices).long().to(values.device)

        tri_map = torch.gather(simplices, dim=0, index=isimplex.repeat([1,3]))
        # print('tri_map max:{}, min{}\n'.format(tri_map.max(),tri_map.min()))
        # print('tri_map_shape: {}, values_shape: {}'.format(tri_map.shape, values.shape))

        value_corr = [torch.gather(values, dim=0, index=tri_map[:,i].
                                    reshape([-1,1]).repeat([1,vdim])) for i in range(3)]
        value_corr = torch.stack(value_corr)
        # print('value_corr_shape: {}'.format(value_corr.shape))
        # print('value_corr have none?: {}'.format(torch.isnan(value_corr).sum()))
        weights = weights.transpose(1,0).unsqueeze(2).repeat([1,1,vdim])
        # print('weights_shape: {}'.format(weights.shape))
        # print('weights have none?: {}'.format(torch.isnan(weights).sum()))
        # print('weights_dtype: {}, value_corr_dtype: {}'.format(weights.dtype, value_corr.dtype))
        out = torch.mul(value_corr, weights).sum(dim=0)
        # print('out_shape: {}'.format(out.shape))
        return out.reshape([self.h, self.w, vdim]).permute(2,0,1)


if __name__=='__main__':
    interp2d = Interp2D(10,10)
    points = torch.rand([10,2])*10
    values = torch.rand([10,2])
    out = interp2d(points,values)
    print('out shape', out.shape)
    print('points\n', points)
    print('values\n', values)
    print(out)
