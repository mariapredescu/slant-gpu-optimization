from nilearn.image import resample_img
import pylab as plt
import nibabel as nb
import numpy as np
import os

nb.Nifti1Header.quaternion_threshold = -1e-06
orig_nii = nb.load("1000_3.nii")

np.round(orig_nii.affine)
orig_nii.shape

orig_nii.header.get_zooms()

plt.imshow(orig_nii.dataobj[:,:,80])

upsampled_nii = resample_img(orig_nii, target_affine=np.eye(3)*0.5, interpolation='nearest')

upsampled_nii.affine

upsampled_nii.shape

plt.imshow(upsampled_nii.dataobj[:,:,200])

nb.save(upsampled_nii, os.path.join('build', 'upsampled_nii.nii.gz')) 