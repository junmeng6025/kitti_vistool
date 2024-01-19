import os
import numpy as np
import seaborn as sns
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from utils.draw_3d_boxes import boxes_to_pcl, boxes_to_img
from skimage import io


def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")


# Draw 3D Box in Point Cloud ================================================================
def draw_3d_boxes_to_pcl(file_id, data_path, output_dir, cls_names,
                         gt_label_path=None, colormap_gt=None,
                         pred_label_path=None, colormap_baseline=None, 
                         ours_label_path=None, colormap_ours=None, save_plot=False):
    # load point clouds
    scan_dir = f'{data_path}/velodyne/{file_id}.bin'
    scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)
    # render the scene
    fig = mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(1920, 1080))
    plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig, color=(0, 0, 0))

    # load labels
    gt_labels = open(f'{gt_label_path}/{file_id}.txt', 'r').readlines()\
        if gt_label_path is not None and os.path.exists(gt_label_path) else None
    boxes_to_pcl(fig, gt_labels, colormap_gt, cls_names, 2)\
        if gt_labels is not None else print(" [pcl] No GT labels loaded.")

    pred_labels = open(f'{pred_label_path}/{file_id}.txt', 'r').readlines()\
        if pred_label_path is not None and os.path.exists(pred_label_path) else None
    boxes_to_pcl(fig, pred_labels, colormap_baseline, cls_names, 2)\
        if pred_labels is not None else print(" [pcl] No Baseline labels loaded.")
    
    ours_labels = open(f'{ours_label_path}/{file_id}.txt', 'r').readlines()\
        if ours_label_path is not None and os.path.exists(ours_label_path) else None
    boxes_to_pcl(fig, ours_labels, colormap_ours, cls_names, 1.5)\
        if ours_labels is not None else print(" [pcl] No Ours labels loaded.")

    # plot scene
    # mlab.view(azimuth=230, distance=50)
    mlab.view(azimuth=-179, elevation=54.0, distance=90.0, roll=90.0)
    if save_plot:
        output_dir = f'{output_dir}/3d_to_pcl'
        create_directory_if_not_exists(output_dir)
        mlab.savefig(filename=f'{output_dir}/kitti_3dbox_to_cloud_{file_id}.png')
        print(f"Scene '{file_id}' saved to {output_dir}.")
    mlab.show()


# Draw 3D Box in Image ==================================================================
def draw_3d_boxes_to_img(file_id, data_path, output_dir, cls_names, 
                         gt_label_path=None, colormap_gt=None,
                         pred_label_path=None, colormap_baseline=None, 
                         ours_label_path=None, colormap_ours=None, save_plot=False):
    # load image
    img = np.array(io.imread(f'{data_path}/image_2/{file_id}.png'), dtype=np.int32)

    # load calibration file
    with open(f'{data_path}/calib/{file_id}.txt', 'r') as f:
        lines = f.readlines()
        cam_calib = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

    fig = plt.figure()
    # draw image
    plt.imshow(img)

    # load labels
    gt_labels = open(f'{gt_label_path}/{file_id}.txt', 'r').readlines()\
        if gt_label_path is not None and os.path.exists(gt_label_path) else None
    boxes_to_img(cam_calib, gt_labels, colormap_gt, cls_names, 2)\
        if gt_labels is not None else print(" [img] No GT labels loaded.")

    pred_labels = open(f'{pred_label_path}/{file_id}.txt', 'r').readlines()\
        if pred_label_path is not None and os.path.exists(pred_label_path) else None
    boxes_to_img(cam_calib, pred_labels, colormap_baseline, cls_names, 2)\
        if pred_labels is not None else print(" [img] No Baseline labels loaded.")

    ours_labels = open(f'{ours_label_path}/{file_id}.txt', 'r').readlines()\
        if ours_label_path is not None and os.path.exists(ours_label_path) else None
    boxes_to_img(cam_calib, ours_labels, colormap_ours, cls_names, 1.5)\
        if ours_labels is not None else print(" [img] No Ours labels loaded.")

    # fig.patch.set_visible(False)
    plt.axis('off')
    plt.tight_layout()
    # plt.get_current_fig_manager().full_screen_toggle()
    
    if save_plot:
        output_dir = f'{output_dir}/3d_to_img'
        create_directory_if_not_exists(output_dir)
        plt.savefig(f'{output_dir}/kitti_3dbox_to_img_{file_id}.png', bbox_inches='tight')
        print(f"Scene '{file_id}' saved to {output_dir}.")
    plt.show()


# TBD =======================================================
def bbox_to_img():
    pass

def pcl_to_img():
    pass

def img_to_pcl():
    pass

def semseg_to_pcl():
    pass


if __name__ == "__main__":
    TAG_OURS = "pvrcnn-relation/2023-09-27_11-17-07/epoch_80"
    TAG_BASLIN = "pvrcnn/2023-09-04_09-52-39/epoch_78"

    DATA_PATH = "/home/jun/datasets/kitti_detection_data/training"
    GT_LABEL = "/home/jun/datasets/kitti_detection_data/training/label_2" # label_path: GT
    OURS_LABEL = "/home/jun/OpenPCDet_vis/from_server/kitti/results_txt/%s/results"%TAG_OURS # label_path: PRED-Ours
    BASLIN_LABEL = "/home/jun/OpenPCDet_vis/from_server/kitti/results_txt/%s/results"%TAG_BASLIN # label_path: PRED-Baseline
    OUTPUT_DIR = "/home/jun/OpenPCDet_vis/scenes/iv_compare"

    SCENE_LS = "/home/jun/OpenPCDet_vis/from_server/kitti/val.txt"
    START_IDX = 645 # default = 0

    TO_PLOT = {"3dbox_to_pcl": True,
               "3dbox_to_img": True,
               "bbox_to_img": False,
               "pcl_to_img": False,
               "img_to_pcl": False,
               "label_to_pcl": False}
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< START >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    file_ids = [x.strip() for x in open(SCENE_LS).readlines()] if os.path.exists(SCENE_LS) else None

    names = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']
    colors = sns.color_palette('Paired', 9 * 2) # Colormap for GT
    colormap_baseline = [(1, 0, 0.25), (1, 0, 0.25), (1, 0, 0.25), (1, 0, 0.25), (1, 0, 0.25), (1, 0, 0.25), (1, 0, 0.25), (1, 0, 0.25)] # format bgr
    colormap_ours = [(0, 0.25, 1), (0, 0.25, 1), (0, 0.25, 1), (1, 0.5, 0), (1, 0.5, 0), (1, 0, 0.5), (0, 0, 0), (0, 0, 0)]

    for file_id in file_ids[START_IDX:]:
        # ============ For a single plot decomment the 'file_id' below ============
        # file_id = "000708"

        # ==================== Plot given scenes in list ====================
        print("\n#%d/%d: -----< Scene %s > -----"%(file_ids.index(file_id), len(file_ids), file_id))
        if TO_PLOT["3dbox_to_pcl"]:
            draw_3d_boxes_to_pcl(file_id, data_path=DATA_PATH, output_dir=OUTPUT_DIR, cls_names=names, 
                gt_label_path=None, colormap_gt=None, 
                pred_label_path=BASLIN_LABEL, colormap_baseline=colormap_baseline, 
                ours_label_path=OURS_LABEL, colormap_ours=colormap_ours)
        if TO_PLOT["3dbox_to_img"]:
            draw_3d_boxes_to_img(file_id, data_path=DATA_PATH, output_dir=OUTPUT_DIR, cls_names=names, 
                gt_label_path=None, colormap_gt=None, 
                pred_label_path=BASLIN_LABEL, colormap_baseline=colormap_baseline, 
                ours_label_path=OURS_LABEL, colormap_ours=colormap_ours)
    