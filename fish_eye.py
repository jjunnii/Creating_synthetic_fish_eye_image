import cv2
import numpy as np
import functions as f
import os, cv2
import numpy as np
import cv2
import argparse


# Auxiliar functions for fish eye image composition
# Computes the parameter PHI of the fish-eye
def cam_system(system, r, f):
    aux = r / f
    if system == 'equiang':
        return aux
    elif system == 'stereo':
        return 2 * np.arctan2(r, 2 * f)
    elif system == 'orth':
        return np.arcsin(aux)
    elif system == 'equisol':
        return 2 * np.arcsin(aux)
    else:
        print('Camera system ERROR')


# Computes focal length for each fish eye sytem
def focal_lenght(system, r_max, phi):
    if system == 'equiang':
        f = r_max / float(phi)
    elif system == 'stereo':
        f = r_max / float((2 * np.tan(phi / 2.0)))
    elif system == 'orth':
        f = r_max / float(np.sin(phi))
    elif system == 'equisol':
        f = r_max / float(np.sin(phi / 2.0))
    else:
        print('Camera system ERROR')
        f = 0
    return f


# Main program
def main(scene, common=[1024, 1024, ['lit'], 0, 1, 'cam_rot.txt', 'R'], specific=['equiang', 185]):


    
    # ----------------------------------------------------------------------------
    # fishe eye image parameters
    final_w = common[0]  # Image resolution: width
    final_h = common[1]  # Image resolution: height
    mode_list = common[2]  # View mode
    init_loc = common[3]  # First location to evaluate
    num_locs = common[4]  # Number of locations
    loc_list = [i + init_loc for i in range(num_locs)]  # List of locations
    rot1 = common[5]
    rot2 = common[6]
    system = specific[0]

    FOV_fish = np.deg2rad(specific[1])
    # ----------------------------------------------------------------------------
    if not cv2.useOptimized():
        print('Turn on the Optimizer')
        cv2.setUseOptimized(True)
    # Geometric parameters
    Nor, Rot = f.load_geom()

    # Camera images - cubemap
    for mode in mode_list:
        for loc in loc_list:
            final = np.zeros((final_h, final_w, 3), np.uint8)
            r, g, b = np.zeros(final_h * final_w), np.zeros(final_h * final_w), np.zeros(final_h * final_w)
            count = 0

            target_gt = args.dataset_dir
            dira = os.path.join(target_gt,'0_out')   #forward
            dirb = os.path.join(target_gt,'3_out')    #back
            dirc = os.path.join(target_gt,'1_out')    #left
            dird = os.path.join(target_gt,'2_out')    #right
            dire = os.path.join(target_gt,'4_out')    #up
            dirk = os.path.join(target_gt,'5_out')    #down
            a11 = sorted(os.listdir(dira))
            b11 = sorted(os.listdir(dirb))
            c11 = sorted(os.listdir(dirc))
            d11 = sorted(os.listdir(dird))
            e11 = sorted(os.listdir(dire))
            q11 = sorted(os.listdir(dirk))
   

            for i1, i2, i3, i4, i5, i6 in zip(a11, b11, c11, d11, e11, q11):

                n1 = cv2.imread(dira +'/' + i1.format(mode, loc, mode, loc))
                n2 = cv2.imread(dirb +'/' + i2.format(mode, loc, mode, loc))
                n3 = cv2.imread(dirc +'/' + i3.format(mode, loc, mode, loc))
                n4 = cv2.imread(dird +'/' + i4.format(mode, loc, mode, loc))
                n5 = cv2.imread(dire +'/' + i5.format(mode, loc, mode, loc))
                n6 = cv2.imread(dirk +'/' + i6.format(mode, loc, mode, loc))
                imagenes = [n1, n2, n3, n4, n5, n6]
                save_dir = args.save_dir
                count = count + 1

                im_h, im_w, ch = imagenes[0].shape
                im_w -= 1
                im_h -= 1

                # Camera parameters
                R_view = f.camera_rotation(rot1, rot2, loc)  # Rotation matrix of the viewer
                # R_world = np.dot(R_cam,R_view)
                r_max = max(final_w / 2, final_h / 2)
                f_fish = focal_lenght(system, r_max, FOV_fish / 2.0)
                FOV = np.pi / 2.0
                fx = (im_w / 2.0) / np.tan(FOV / 2.0)
                fy = (im_h / 2.0) / np.tan(FOV / 2.0)
                K = np.array([[fx, 0, im_w / 2.0], [0, fy, im_h / 2.0], [0, 0, 1]])

                print('making process...')
                # Pixel mapping
                x_0, y_0 = final_w / 2.0, final_h / 2.0
                x, y = np.meshgrid(np.arange(final_w), np.arange(final_h))
                r_hat = np.sqrt(np.square(x_0 - x) + np.square(y - y_0))
                out = r_hat > r_max
                out = out.reshape(1, r_hat.size)
                theta = np.arctan2(x - x_0, y - y_0)
                phi = cam_system(system, r_hat, f_fish)
                ct, st, cp, sp = np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi)
                vec = np.array([(sp * st), (sp * ct), (cp)]).reshape(3, final_w * final_h)
                v_abs = np.dot(R_view, vec)
                img_index = f.get_index(v_abs)
                for i in range(img_index.size):
                    if out[0, i]:
                        continue
                    n, imagen, R = Nor[img_index[i]], imagenes[img_index[i]], Rot[img_index[i]]
                    p_x, p_y = f.get_pixel(v_abs[:, i], R, K)
                    color = imagen[p_y, p_x]
                    r[i], g[i], b[i] = color[0:3]

                final = cv2.merge((r, g, b)).reshape(final_h, final_w, 3)

                cv2.imwrite(os.path.join(save_dir, f"{str(count).zfill(6)}.png"), final)


if __name__ =="__main__":
    scene = 'jjun'
    parser = argparse.ArgumentParser(description='Making fisheyeimg.')
    parser.add_argument('--dataset_dir', type=str, default='./dataset')
    parser.add_argument('--save_dir', type=str, default='./dataset/fisheye')
    parser.add_argument('--Fov', type=int, default=185)
    args = parser.parse_args()
    main(scene)



