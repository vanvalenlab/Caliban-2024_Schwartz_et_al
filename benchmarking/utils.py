import os

import numpy as np
from tifffile import imwrite

from deepcell_tracking.isbi_utils import trk_to_isbi
from deepcell_tracking.utils import contig_tracks


def find_zero_padding(X):
    """Remove zero padding to avoid adverse effects on model performance"""
    # Calculate position of padding based on first frame
    # Assume that padding is in blocks on the edges of image
    good_rows = np.where(X[0].any(axis=0))[0]
    good_cols = np.where(X[0].any(axis=1))[0]

    slc = (
        slice(None),
        slice(good_cols[0], good_cols[-1] + 1),
        slice(good_rows[0], good_rows[-1] + 1),
        slice(None)
    )

    return slc


def save_ctc_raw(exp_dir, batch, X):
    raw_dir = os.path.join(exp_dir, '{:03}'.format(batch))
    
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
        
    # Save each frame as a tiff file
    for i in range(X.shape[0]):
        imwrite(os.path.join(raw_dir, 't{:03}.tif'.format(i)), X[i])
        

def save_ctc_gt(exp_dir, batch, y, lineage):
    gt_dir = os.path.join(exp_dir, '{:03}_GT'.format(batch))
    seg_dir = os.path.join(gt_dir, 'SEG')
    tra_dir = os.path.join(gt_dir, 'TRA')
    
    for d in [gt_dir, seg_dir, tra_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    # Save lineage to isbi txt
    df = trk_to_isbi(lineage)
    df.to_csv(os.path.join(tra_dir, 'man_track.txt'), sep=' ', header=False, index=False)
    
    # Save each frame as a tiff file
    for i in range(y.shape[0]):
        imwrite(os.path.join(seg_dir, 'man_seg{:03}.tif'.format(i)), y[i].astype('uint16'))
        imwrite(os.path.join(tra_dir, 'man_track{:03}.tif'.format(i)), y[i].astype('uint16'))
        
            
def save_ctc_res(exp_dir, batch, y, lineage=None, seg=False):
    if seg:
        name = '{:03}_SEG_RES'.format(batch)
    else:
        name = '{:03}_RES'.format(batch)
    res_dir = os.path.join(exp_dir, name)
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
        
    # Save lineage to isbi txt
    if lineage:
        df = trk_to_isbi(lineage)
        df.to_csv(os.path.join(res_dir, 'res_track.txt'), sep=' ', header=False, index=False)
    
    # Save each frame as a tiff file
    for i in range(y.shape[0]):
        imwrite(os.path.join(res_dir, 'mask{:03}.tif'.format(i)), y[i].astype('uint16'))
        
        
def convert_to_contiguous(y, lineage):
    done_labels = []
    while set(done_labels) != set(lineage.keys()):
        leftover_labels = [l for l in lineage.keys() if l not in done_labels]
        for label in leftover_labels:
            lineage, y = contig_tracks(label, lineage, y)
            done_labels.append(label)
    
    return y, lineage
