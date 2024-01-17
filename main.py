import os
import numpy as np
import seaborn as sns
import mayavi.mlab as mlab

import matplotlib.pyplot as plt

from skimage import io
from matplotlib.lines import Line2D

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        print(f"Directory '{path}' already exists.")

# Draw 3D Box in Point Cloud ================================================================
def three_d_box_to_pcl(file_id, data_path, output_dir, gt_label_path=None, colormap_gt=None,
                       pred_label_path=None, colormap_baseline=None, ours_label_path=None, colormap_ours=None, save_plot=False):
    # load point clouds
    scan_dir = f'{data_path}/velodyne/{file_id}.bin'
    scan = np.fromfile(scan_dir, dtype=np.float32).reshape(-1, 4)

    # load labels
    # label_dir = f'{label_path}/{file_id}.txt'
    # with open(label_dir, 'r') as f:
    #     labels = f.readlines()

    fig = mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(1920, 1080))
    # draw point cloud
    plot = mlab.points3d(scan[:, 0], scan[:, 1], scan[:, 2], mode="point", figure=fig, color=(0, 0, 0))

    def draw_boxes(labels, fig, colors, lin_wid):
        for line in labels:
            line = line.split()
            lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line[:15] # for pred label, ignore 'score'
            h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
            # if lab != 'DontCare':
            if lab in ['Car', 'Van', 'Truck']:
                x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
                z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

                # transform the 3d bbox from object coordiante to camera_0 coordinate
                R = np.array([[np.cos(rot), 0, np.sin(rot)],
                                [0, 1, 0],
                                [-np.sin(rot), 0, np.cos(rot)]])
                corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

                # transform the 3d bbox from camera_0 coordinate to velodyne coordinate
                corners_3d = corners_3d[:, [2, 0, 1]] * np.array([[1, -1, -1]])


                def draw(p1, p2, front=1):
                    # mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    #             color=colors[names.index(lab) * 2 + front], tube_radius=None, line_width=1, figure=fig)
                    mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                color=colors[names.index(lab)], tube_radius=None, line_width=lin_wid, figure=fig)

                # draw the upper 4 horizontal lines
                draw(corners_3d[0], corners_3d[1], 0)  # front = 0 for the front lines
                draw(corners_3d[1], corners_3d[2])
                draw(corners_3d[2], corners_3d[3])
                draw(corners_3d[3], corners_3d[0])

                # draw the lower 4 horizontal lines
                draw(corners_3d[4], corners_3d[5], 0)
                draw(corners_3d[5], corners_3d[6])
                draw(corners_3d[6], corners_3d[7])
                draw(corners_3d[7], corners_3d[4])

                # draw the 4 vertical lines
                draw(corners_3d[4], corners_3d[0], 0)
                draw(corners_3d[5], corners_3d[1], 0)
                draw(corners_3d[6], corners_3d[2])
                draw(corners_3d[7], corners_3d[3])

    # load labels
    gt_labels = open(f'{gt_label_path}/{file_id}.txt', 'r').readlines()\
        if gt_label_path is not None and os.path.exists(gt_label_path) else None
    draw_boxes(gt_labels, fig, colormap_gt, 2)\
        if gt_labels is not None else print(" [pcl] No GT labels loaded.")

    pred_labels = open(f'{pred_label_path}/{file_id}.txt', 'r').readlines()\
        if pred_label_path is not None and os.path.exists(pred_label_path) else None
    draw_boxes(pred_labels, fig, colormap_baseline, 2)\
        if pred_labels is not None else print(" [pcl] No Baseline labels loaded.")
    
    ours_labels = open(f'{ours_label_path}/{file_id}.txt', 'r').readlines()\
        if ours_label_path is not None and os.path.exists(ours_label_path) else None
    draw_boxes(ours_labels, fig, colormap_ours, 1.5)\
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
def three_d_box_to_img(file_id, data_path, output_dir, gt_label_path=None, colormap_gt=None,
                       pred_label_path=None, colormap_baseline=None, ours_label_path=None, colormap_ours=None, save_plot=False):
    # load image
    img = np.array(io.imread(f'{data_path}/image_2/{file_id}.png'), dtype=np.int32)

    # load labels
    # with open(f'{label_path}/{file_id}.txt', 'r') as f:
    #     labels = f.readlines()

    # load calibration file
    with open(f'{data_path}/calib/{file_id}.txt', 'r') as f:
        lines = f.readlines()
        P2 = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

    fig = plt.figure()
    # draw image
    plt.imshow(img)

    def draw_boxes(labels, colors, lin_wid):
        for line in labels:
            line = line.split()
            lab, _, _, _, _, _, _, _, h, w, l, x, y, z, rot = line[:15] # for pred label, ignore 'score'
            h, w, l, x, y, z, rot = map(float, [h, w, l, x, y, z, rot])
            # if lab != 'DontCare':
            if lab in ['Car', 'Van', 'Truck']:
                x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
                z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                corners_3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)

                # transform the 3d bbox from object coordiante to camera_0 coordinate
                R = np.array([[np.cos(rot), 0, np.sin(rot)],
                                [0, 1, 0],
                                [-np.sin(rot), 0, np.cos(rot)]])
                corners_3d = np.dot(R, corners_3d).T + np.array([x, y, z])

                # transform the 3d bbox from camera_0 coordinate to camera_x image
                corners_3d_hom = np.concatenate((corners_3d, np.ones((8, 1))), axis=1)
                corners_img = np.matmul(corners_3d_hom, P2.T)
                corners_img = corners_img[:, :2] / corners_img[:, 2][:, None]


                def line(p1, p2, front=1):
                    # line = Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=colors[names.index(lab) * 2 + front])
                    line = Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=colors[names.index(lab)])
                    line.set_linewidth(lin_wid)
                    plt.gca().add_line(line)


                # draw the upper 4 horizontal lines
                line(corners_img[0], corners_img[1], 0)  # front = 0 for the front lines
                line(corners_img[1], corners_img[2])
                line(corners_img[2], corners_img[3])
                line(corners_img[3], corners_img[0])

                # draw the lower 4 horizontal lines
                line(corners_img[4], corners_img[5], 0)
                line(corners_img[5], corners_img[6])
                line(corners_img[6], corners_img[7])
                line(corners_img[7], corners_img[4])

                # draw the 4 vertical lines
                line(corners_img[4], corners_img[0], 0)
                line(corners_img[5], corners_img[1], 0)
                line(corners_img[6], corners_img[2])
                line(corners_img[7], corners_img[3])

    # load labels
    gt_labels = open(f'{gt_label_path}/{file_id}.txt', 'r').readlines()\
        if gt_label_path is not None and os.path.exists(gt_label_path) else None
    draw_boxes(gt_labels, colormap_gt, 2) if gt_labels is not None else print(" [img] No GT labels loaded.")

    pred_labels = open(f'{pred_label_path}/{file_id}.txt', 'r').readlines()\
        if pred_label_path is not None and os.path.exists(pred_label_path) else None
    draw_boxes(pred_labels, colormap_baseline, 2) if pred_labels is not None else print(" [img] No Baseline labels loaded.")

    ours_labels = open(f'{ours_label_path}/{file_id}.txt', 'r').readlines()\
        if ours_label_path is not None and os.path.exists(ours_label_path) else None
    draw_boxes(ours_labels, colormap_ours, 1.5) if ours_labels is not None else print(" [img] No Ours labels loaded.")

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


def bbox_to_img():
    pass

def pcl_to_img():
    pass

def img_to_pcl():
    pass

def label_to_pcl():
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

    TO_PLOT = {"3dbox_to_pcl": False,
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
            three_d_box_to_pcl(file_id, data_path=DATA_PATH, output_dir=OUTPUT_DIR, 
                               gt_label_path=None, colormap_gt=None, 
                               pred_label_path=BASLIN_LABEL, colormap_baseline=colormap_baseline, 
                               ours_label_path=OURS_LABEL, colormap_ours=colormap_ours)
        if TO_PLOT["3dbox_to_img"]:
            three_d_box_to_img(file_id, data_path=DATA_PATH, 
                               output_dir=OUTPUT_DIR, gt_label_path=None, colormap_gt=None, 
                               pred_label_path=BASLIN_LABEL, colormap_baseline=colormap_baseline, 
                               ours_label_path=OURS_LABEL, colormap_ours=colormap_ours)
    