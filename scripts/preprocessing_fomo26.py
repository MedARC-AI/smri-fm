# preprocessing script for fomo26 tasks
# `build_fomo26_submission.sh` copies this file into the submission image file

import numpy as np, torch, torch.nn.functional as F, nibabel as nib
from torch import Tensor
from fastcore.basics import store_attr
from fasttransform import Transform,Pipeline
from nibabel import Nifti1Image
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation

class Reorient(Transform):
    'Approximately reorient to RAS by applying axis-swaps and axis-flips only'
    def encodes(self, x:Nifti1Image):
        out = nib.as_closest_canonical(x)
        in_ornt, out_ornt  = io_orientation(x.affine), io_orientation(out.affine)
        self.affine = x.affine
        self.inv = ornt_transform(out_ornt, in_ornt)
        return out
    def decodes(self, x:Nifti1Image): return Nifti1Image( dataobj=apply_orientation(x.get_fdata(), self.inv), affine=self.affine )

class Resample(Transform):
    def __init__(self, target_spacing=(1.,1.,1.), fwd_mode='trilinear', bwd_mode='nearest', threshold=0.05): store_attr()
    def encodes(self, x:Nifti1Image):
        is_,tgt = np.array(x.header.get_zooms()), np.array(self.target_spacing)
        if np.abs(is_-tgt).max() <= self.threshold:
            self.orig_shape,self.orig_affine = None,None
            return x
        self.orig_shape,self.orig_affine = x.shape, x.affine.copy()
        data = F.interpolate(Tensor(x.get_fdata())[None,None], scale_factor=tuple(is_/tgt), mode=self.fwd_mode).squeeze(0,1)
        new_aff = x.affine.copy()
        new_aff[:3,:3] *= tgt/is_
        return Nifti1Image(data.numpy(), new_aff)
    def decodes(self, x:Nifti1Image):
        if self.orig_shape is None: return x
        data = F.interpolate(Tensor(x.get_fdata())[None,None], size=tuple(self.orig_shape), mode=self.bwd_mode).squeeze(0,1)
        return Nifti1Image(data.numpy(), self.orig_affine)

class Unwrap(Transform):
    'Extract volume tensor from nifti'
    def encodes(self, x:Nifti1Image):
        self.affine = x.affine.copy()
        return Tensor(x.get_fdata())
    def decodes(self, x:Tensor):
        return Nifti1Image(x.numpy(), self.affine)

class ToZXY(Transform):
    'Transpose (X,Y,Z) -> (Z,Y,X) C-order'
    def encodes(self, x:Tensor): return x.permute(2,1,0).contiguous()
    def decodes(self, x:Tensor): return x.permute(2,1,0).contiguous()

class CenterPad(Transform):
    def __init__(self, tgt_shape, crop=False): store_attr()
    def _padcrop(self, x, tgt, crop):
        pads = []
        for s,s_ in zip(x.shape, tgt):
            d = s_-s
            if not crop: d = max(0,d)
            pads.extend([d//2, d - d//2])
        return F.pad(x, pads[::-1])
    def encodes(self, x:Tensor):
        self.orig_shape = x.shape
        return self._padcrop(x, self.tgt_shape, self.crop)
    def decodes(self, x:Tensor):
        return self._padcrop(x, self.orig_shape, crop=True)

class Normalize(Transform):
    'Normalize values to standard Gaussian, ignoring mask'
    def encodes(self, x:Tensor):
        self.mask = x>x.mean() # Crude but cheap brain mask
        xm = x[self.mask]
        self.mean,self.std = xm.mean(), xm.std(correction=0).clamp_min(1e-6)
        return torch.where(self.mask, (x-self.mean)/self.std, 0.)

class AddChanelDim(Transform):
    def encodes(self, x:Tensor): return x[None,:]

def preproc_pipe(im_sz, crop): return Pipeline([ Reorient(), Resample(), Unwrap(), ToZXY(), CenterPad(im_sz, crop), Normalize(), AddChanelDim() ])
