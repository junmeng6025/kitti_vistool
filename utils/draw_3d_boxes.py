import numpy as np
import mayavi.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def boxes_to_pcl(fig, labels, colors, cls_names, lin_wid, front_mod=None):
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

            if front_mod == "light_colored":
                def draw(p1, p2, front=1):
                    mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                color=colors[cls_names.index(lab) * 2 + front], 
                                tube_radius=None, line_width=lin_wid, figure=fig)
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

            else:
                def draw(p1, p2):
                        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                    color=colors[cls_names.index(lab)], 
                                    tube_radius=None, line_width=lin_wid, figure=fig)
                # draw the upper 4 horizontal lines
                draw(corners_3d[0], corners_3d[1])
                draw(corners_3d[1], corners_3d[2])
                draw(corners_3d[2], corners_3d[3])
                draw(corners_3d[3], corners_3d[0])

                # draw the lower 4 horizontal lines
                draw(corners_3d[4], corners_3d[5])
                draw(corners_3d[5], corners_3d[6])
                draw(corners_3d[6], corners_3d[7])
                draw(corners_3d[7], corners_3d[4])

                # draw the 4 vertical lines
                draw(corners_3d[4], corners_3d[0])
                draw(corners_3d[5], corners_3d[1])
                draw(corners_3d[6], corners_3d[2])
                draw(corners_3d[7], corners_3d[3])

                if front_mod == "cross_face":
                # draw the cross for front face
                    draw(corners_3d[4], corners_3d[1])
                    draw(corners_3d[5], corners_3d[0])


def boxes_to_img(cam_calib, labels, colors, cls_names, lin_wid, front_mod="cross_face"):
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
            corners_img = np.matmul(corners_3d_hom, cam_calib.T)
            corners_img = corners_img[:, :2] / corners_img[:, 2][:, None]

            if front_mod == "light_colored":
                def line(p1, p2, front=1):
                    line = Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=colors[cls_names.index(lab) * 2 + front])
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
            
            else:
                def line(p1, p2):
                    line = Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=colors[cls_names.index(lab)])
                    line.set_linewidth(lin_wid)
                    plt.gca().add_line(line)
                
                # draw the upper 4 horizontal lines
                line(corners_img[0], corners_img[1])  # front = 0 for the front lines
                line(corners_img[1], corners_img[2])
                line(corners_img[2], corners_img[3])
                line(corners_img[3], corners_img[0])

                # draw the lower 4 horizontal lines
                line(corners_img[4], corners_img[5])
                line(corners_img[5], corners_img[6])
                line(corners_img[6], corners_img[7])
                line(corners_img[7], corners_img[4])

                # draw the 4 vertical lines
                line(corners_img[4], corners_img[0])
                line(corners_img[5], corners_img[1])
                line(corners_img[6], corners_img[2])
                line(corners_img[7], corners_img[3])

                if front_mod == None:
                    line(corners_img[4], corners_img[1])
                    line(corners_img[5], corners_img[0])