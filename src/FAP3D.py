import dlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from argparse import ArgumentParser
from ast import literal_eval
from glob import glob
from icp import icp
from math import degrees
from PRNet.api import PRN
from PRNet.utils.write import write_obj_with_colors
from scipy.io import loadmat
from scipy.spatial import distance
from skimage.io import imread
from skimage.transform import rescale
from sys import stdout

def main(args):
    if args.saveOutput:
        args.calculateAllPointsError = True
        args.calculateKeypointsError = False

    # init CUDA
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0" # ID of GPU, -1 for CPU

    # init PRN
    prn = PRN(is_dlib = True)

    # init ground truth model info
    mat_file_content = loadmat('./../data/model_info.mat')
    ground_truth_keypoint_indices = np.array(mat_file_content['keypoints'])[0]
    all_vertices_indices = np.loadtxt('./../data/face_ind.txt').astype(np.int32)

    # validate image folders
    image_folder = args.inputFolder
    ground_truth_folder = args.groundTruthFolder
    assert os.path.exists(image_folder) and os.path.exists(ground_truth_folder)

    # get front and angled image paths with mathcing face angle
    print('\nextracting image data from input folder...')
    image_path_dict = get_300W_front_and_profile_image_paths_dict(image_folder, image_angle2=args.sideAngle)

    # sample k random image ids from image path dictionary
    sampled_face_ids = random.sample(image_path_dict.keys(), k = args.k)
    sampled_face_ids_length = len(sampled_face_ids)

    not_outlier = 0
    individual_error_values = []
    # iterate over each image pair and find error
    for i, face_id in enumerate(sampled_face_ids):
        print('\nface id:',face_id)
        print('', sampled_face_ids_length - i,'face(s) left..')

        # read images
        image1_path = image_path_dict[face_id][0] # front image
        image2_path = image_path_dict[face_id][1] # angled/profile image
        image1 = get_image(image1_path)
        image2 = get_image(image2_path)
        assert image1.shape == image2.shape
        h = image1.shape[0] # height

        # get position maps from images
        print('getting position map 2...')
        position_map2 = get_pos_from_image(image2, prn)
        if position_map2 is None:
            print('could not detect face')
            continue
        print('getting position map 1...')
        position_map1 = get_pos_from_image(image1, prn)
        if position_map1 is None:
            print('could not detect face')
            continue

        # get vertices from position maps
        #start = time.time()
        raw_vertices1, vertices1 = get_vertices_from_pos(h, position_map1, prn)
        #end = time.time()
        #print('single pass:', end - start)
        raw_vertices2, vertices2 = get_vertices_from_pos(h, position_map2, prn)

        # get keypoints from position maps 
        keypoints1 = get_keypoints_from_pos(h, position_map1, prn)
        keypoints2 = get_keypoints_from_pos(h, position_map2, prn)

        # find ground truth keypoints and vertices for face
        keypoints_ground_truth, raw_vertices_ground_truth, vertices_ground_truth = get_300W_ground_truth_keypoints_and_vertices(ground_truth_folder, face_id, ground_truth_keypoint_indices, all_vertices_indices)
        if vertices_ground_truth is None or keypoints_ground_truth is None:
            print('could not find ground truth data')
            continue

        #get initial alignment matrices, found empirically
        init_align1 = get_initial_alignment_trans_matrix_front()
        init_align2 = get_initial_alignment_trans_matrix_front()
        if (args.sideAngle > 45 and args.sideAngle < 90):
            init_align2 = get_initial_alignment_trans_matrix_left()
        elif (args.sideAngle < -45 and args.sideAngle > -90):
            init_align2 = get_initial_alignment_trans_matrix_right()
        
        # align keypoints and vertices with ground truth
        if args.calculateKeypointsError:
            _,interocular_distance, keypoints1 = align_vertices(keypoints1, keypoints_ground_truth, init_align = init_align1)
            _,interocular_distance, keypoints2 = align_vertices(keypoints2, keypoints_ground_truth, init_align = init_align2)
        elif args.calculateAllPointsError:
            _,interocular_distance, keypoints1, vertices1 = align_keypoints_and_vertices(keypoints1, keypoints_ground_truth, vertices1, init_align = init_align1)
            _,interocular_distance, keypoints2, vertices2 = align_keypoints_and_vertices(keypoints2, keypoints_ground_truth, vertices2, init_align = init_align2)

        # calculate and align average of predicted keypoints and vertices
        if args.calculateKeypointsError:
            keypoints_average = np.array([keypoints2, keypoints1]).mean(axis=0)
            _,interocular_distance, keypoints_average = align_vertices(keypoints_average, keypoints_ground_truth)
            keypoints_experimental_average = get_weighted_average(keypoints2, keypoints1)
            _,interocular_distance, keypoints_experimental_average = align_vertices(keypoints_experimental_average, keypoints_ground_truth)
        elif args.calculateAllPointsError:
            vertices_average = np.array([vertices1, vertices2]).mean(axis = 0)
            _,interocular_distance, vertices_average = align_vertices(vertices_average, vertices_ground_truth)
        
        # calculate ground truth error for front, side and average
        if args.calculateKeypointsError:
            nse_1 = normalized_squared_error(keypoints1, keypoints_ground_truth, interocular_distance)
            nse_2 = normalized_squared_error(keypoints2, keypoints_ground_truth, interocular_distance)
            nse_a = normalized_squared_error(keypoints_average, keypoints_ground_truth, interocular_distance)
            #nse_ea = normalized_squared_error(keypoints_experimental_average, keypoints_ground_truth, interocular_distance)
            
        if args.calculateAllPointsError:
            nse_1 = normalized_squared_error(vertices1, vertices_ground_truth, interocular_distance)
            nse_2 = normalized_squared_error(vertices2, vertices_ground_truth, interocular_distance)
            nse_a = normalized_squared_error(vertices_average, vertices_ground_truth, interocular_distance)

        # check if result is outlier
        if (np.mean(nse_2) < np.mean(nse_1)*10) and (np.mean(nse_1) < np.mean(nse_2)*10):
            #individual_error_values.append([nse_1, nse_2, nse_a, nse_ea])
            individual_error_values.append([nse_1, nse_2, nse_a])
            print('success calculating error')
            not_outlier += 1
        else:
            print('bad ICP fit or outlier prediciton')

        #save output
        if args.saveOutput:
            colors = prn.get_colors(image1, raw_vertices1)
            plt.imsave('results/' + face_id + 'front.jpg', image1)
            plt.imsave('results/' + face_id + 'side.jpg', image2)
            write_obj_with_colors('results/' + face_id + '_average.obj', vertices1, prn.triangles, colors)
            np.savetxt('results/' + face_id + '_front.txt', vertices1, delimiter=';')
            np.savetxt('results/' + face_id + '_side.txt', vertices2, delimiter=';')
            np.savetxt('results/' + face_id + '_average.txt', vertices_average, delimiter=';')
            np.savetxt('results/' + face_id +'_ground_truth.txt', raw_vertices_ground_truth, delimiter=';')

        print('NME front face:', np.average(nse_1, axis=0))
        print('NME side face:', np.average(nse_2, axis=0))
        print('NME average:', np.average(nse_a, axis=0))
        #print('NME exp average:', np.average(nse_ea, axis=0))

    print((sampled_face_ids_length-not_outlier)*100/sampled_face_ids_length,'% of faces with no detected face, outlier prediciton or bad ICP fit')
    
    # plot results 
    NME_values = np.average(individual_error_values, axis=0)
    print('Mean NME front face: ', np.average(NME_values[0]))
    print('Mean NME side face: ', np.average(NME_values[1]))
    print('Mean NME average: ', np.average(NME_values[2]))
    
    plot_hist_of_distances(individual_error_values)
    plot_individual_keypoint_distances(individual_error_values)
    plot_ced_curves(individual_error_values)

def get_initial_alignment_trans_matrix_front():
    init_align = np.zeros((4,4))
    init_align[0] = np.array([1,   0   ,0   ,   0])
    init_align[1] = np.array([0   ,1   ,0   ,   0])
    init_align[2] = np.array([0   ,0   ,1   ,-200])
    init_align[3] = np.array([0   ,0   ,0   ,   1])
    return init_align

def get_initial_alignment_trans_matrix_left():
    init_align = np.zeros((4,4))
    init_align[0] = np.array([0.3 ,0   ,0.9 , 50])
    init_align[1] = np.array([0   ,1   ,0   ,  0])
    init_align[2] = np.array([-0.9,0   ,0.3 ,200])
    init_align[3] = np.array([0   ,0   ,0   ,  1])
    return init_align

def get_initial_alignment_trans_matrix_right():
    init_align = np.zeros((4,4))
    init_align[0] = np.array([0.1 ,0   ,-0.9, 200])
    init_align[1] = np.array([0   ,1   ,0   ,  0])
    init_align[2] = np.array([0.9 ,0   ,0.1 ,-200])
    init_align[3] = np.array([0   ,0   ,0   ,  1])
    return init_align

def get_weighted_average(keypoints2, keypoints1):
    weights = np.zeros((68,2))
    keypoints_experimental_average = np.zeros((68,3))
    target_keypoints = np.array([0,8,12,16,27,28,29,30,34,53,54,64]) # found from test results

    for i,_ in enumerate(weights):
        if i in target_keypoints:
            weights[i] = [1,0]
        else:
            weights[i] = [0.5, 0.5]
    for i, weight in enumerate(weights):
        keypoints_experimental_average[i] = np.average([keypoints2[i], keypoints1[i]], axis = 0, weights=weights[i])

    return keypoints_experimental_average

def plot_hist_of_distances(individual_error_values):
    hist_values = np.average(np.array(individual_error_values).T, axis=0).T
    bins = np.arange(0,0.1,0.005)
    width = 0.002
    fig, ax = plt.subplots()
    plt.hist(hist_values, bins, histtype='bar', width = width, label = ['front image', 'side image', 'average'], color=['blue', 'lightblue', 'm'])
    #plt.hist(hist_values, bins, histtype='bar', width = width, label = ['front image', 'side image'], color=['blue', 'lightblue'])
    ax.set_ylabel('number of images')
    ax.set_xlabel('NME')
    plt.legend(loc='upper right')
    plt.show()

#cumulative error distribution curve
def plot_ced_curves(individual_error_values):
    hist_values = np.average(np.array(individual_error_values).T, axis=0).T
    bins = np.arange(0,0.1,0.0001)
    fig, ax = plt.subplots()
    plt.hist(hist_values, bins, histtype='step', linewidth= 1.5, cumulative=True, density=True, label=['front image','side image', 'average'] , color = ['blue','lightblue', 'm'])
    #plt.hist(hist_values, bins, histtype='step', linewidth= 1.5, cumulative=True, density=True, label=['front image','side image'] , color = ['blue','lightblue'])
    plt.xticks(bins*100)
    ax.set_ylabel('percentage of images')
    ax.set_xlabel('NME')
    plt.grid(b=True, alpha= 0.5)
    plt.margins(x=0, y=0.05)
    plt.legend(loc='lower right')
    plt.show()
    return

def plot_individual_keypoint_distances(individual_error_values):
    averaged_error_values = np.average(individual_error_values, axis=0)

    error_values_compared_to_first = averaged_error_values.copy()
    for i, averaged_error in enumerate(averaged_error_values):
        error_values_compared_to_first[i] = np.subtract(averaged_error, averaged_error_values[0])
    
    if (np.array(individual_error_values).shape[2] > 500):
        print('too many individual points to show data for each point')
        return
    x = np.arange(0,error_values_compared_to_first.shape[1], 1) + 1
    fig, axes = plt.subplots(nrows=2, ncols=1)
    ax0, ax1 = axes.flatten()
    width = 0.3

    front_vals_0 = ax0.bar(x - width, height=averaged_error_values[0], width = width, label = 'front', color = 'blue')
    side_vals_0 = ax0.bar(x , height=averaged_error_values[1], width = width, label = 'side', color= 'lightblue')
    avg_vals_0 = ax0.bar(x + width, height=averaged_error_values[2], width = width, label = 'average', color= 'm')
    front_vals_1 = ax1.bar(x - width, height=error_values_compared_to_first[0], width = width, label = 'front', color = 'blue')
    side_vals_1 = ax1.bar(x, height=error_values_compared_to_first[1], width = width, label = 'side', color= 'lightblue')
    avg_vals_1 = ax1.bar(x + width, height=error_values_compared_to_first[2], width = width, label = 'average', color= 'm')
    ax0.legend(loc='upper right')
    ax0.set_ylabel('NME')
    ax0.set_xlabel('keypoint id')
    ax0.margins(0)

    ax1.legend(loc='upper right')
    ax1.set_ylabel('NME difference from front')
    ax1.set_xlabel('keypoint id')
    ax1.margins(0)

    plt.setp(axes, xticks=x)
    fig.tight_layout()
    plt.show()

def align_keypoints_and_vertices(keypoints1, keypoints2, vertices1, init_align = None):
    # find scale factor and scale keypoints and vertices to correct size
    scale_factor, interocular_distance = find_scale(keypoints1, keypoints2)
    keypoints1 = keypoints1*scale_factor
    vertices1 = vertices1*scale_factor

    # find transformation matrix to map keypoints1 to keypoints2 using icp algorithm
    T12, distances12, i12 = icp(keypoints1, keypoints2, max_iterations = 50, tolerance= 0.001, init_pose = init_align)

    # apply transformation to keypoints and vertices
    keypoints1_aligned_to_2 = apply_homogenous_transformation_matrix_to_3d_vertices(T12, keypoints1)
    vertices1_aligned_to_2 = apply_homogenous_transformation_matrix_to_3d_vertices(T12, vertices1)
    return T12, interocular_distance, keypoints1_aligned_to_2, vertices1_aligned_to_2

def align_vertices(keypoints1, keypoints2, init_align = None):
    # align keypionts1 to keypoints2
    # find scale factor and scale keypoints to same size
    scale_factor, interocular_distance = find_scale(keypoints1, keypoints2)
    keypoints1 = keypoints1*scale_factor

    # find transformation matrix to map keypoints1 to keypoints2 using icp algorithm
    T12, distances12, i12 = icp(keypoints1, keypoints2, max_iterations = 50, tolerance= 0.001, init_pose = init_align)
    
    # apply transformation to keypoints
    keypoints1_aligned_to_2 = apply_homogenous_transformation_matrix_to_3d_vertices(T12, keypoints1)
    return T12, interocular_distance, keypoints1_aligned_to_2

def find_scale(keypoints1, keypoints2, left_outer_eye_index = 36, right_outer_eye_index = 45):
    # find scale factor and scale keypoints and vertices to correct size
    # keypoints index convention found in 300W-lp dataset
    interocular_dist_1 = distance.euclidean(keypoints1[left_outer_eye_index],keypoints1[right_outer_eye_index])
    interocular_distance = distance.euclidean(keypoints2[left_outer_eye_index],keypoints2[right_outer_eye_index])
    scale_factor = interocular_distance / interocular_dist_1
    return scale_factor, interocular_distance

def get_keypoints_from_pos(h, position_map, prn):
    keypoints = prn.get_landmarks(position_map)
    keypoints[:,1] = h - 1 - keypoints[:,1]
    return keypoints

def get_300W_front_and_profile_image_paths_dict(image_folder, image_angle1 = 0, image_angle2 = 70):
    # populate image path dictionary
    # images must be in correct format: HELEN_232194_1_0.jpg, HELEN_232194_1_0.mat, IBUG_2908549_1_4.jpg etc...
    # { "face id" : ["image path front", "image path profile"]}
    image_path_dict = dict()
    image_paths = glob(image_folder + '\\*.jpg')
    num_paths = len(image_paths)
    for i, file_name in enumerate(image_paths):
        stdout.write("\rfile %i out of %i" % (i, num_paths))
        stdout.flush()
        file_string_split = file_name.split('_')
        face_id = file_string_split[-3] + '_' + file_string_split[-2]
        
        # pose param has jaw angle in radians
        # accept angles in the range +/-20  and +/-5
        ground_turth_mat_file = loadmat(file_name[:-4] + '.mat')
        jaw_angle = round(degrees(ground_turth_mat_file['Pose_Para'][0][1]),0)
        angle_1_range = np.arange(image_angle1-20, image_angle1+20) #front
        angle_2_range = np.arange(image_angle2-5, image_angle2+5) #side

        if image_path_dict.get(face_id) == None:
            if jaw_angle in angle_1_range:
                image_path_dict[face_id] = [file_name, None]
            elif jaw_angle in angle_2_range:
                image_path_dict[face_id] = [None, file_name]
        else:
            if jaw_angle in angle_1_range:
                image_path_dict[face_id] = [file_name, image_path_dict.get(face_id)[1]]
            elif jaw_angle in angle_2_range:
                image_path_dict[face_id] = [image_path_dict.get(face_id)[0], file_name]
    image_path_dict = remove_single_element_keys(image_path_dict)
    return image_path_dict

def remove_single_element_keys(image_path_dict):
    keys = 0
    rem_keys = 0
    # remove images with only one available angle
    for key in list(image_path_dict):
        keys += 1
        if None in image_path_dict[key]:
            del image_path_dict[key]
            rem_keys += 1
    #print('removed: ', rem_keys, ' keys without angle out of:', keys)
    return image_path_dict

def apply_homogenous_transformation_matrix_to_3d_vertices(T, V):
    V_a = np.ones((V.shape[0], V.shape[1] + 1))
    V_a[:,:3] = V.copy()
    V = np.dot(T,V_a.T).T
    return V[:,:3]

def get_image(img_path):
    img = imread(img_path)
    [h, w, c] = img.shape
    if c>3:
        img = img[:,:,:3]
    return img

def get_vertices_from_pos(h, pos, prn):
    raw_vertices = prn.get_vertices(pos)
    vertices = raw_vertices.copy()
    vertices[:,1] = h - 1 - vertices[:,1]
    return raw_vertices, vertices
    
def get_pos_from_image(image, prn):
    #crop image if bigger than 1000 in width or height
    max_size = max(image.shape[0], image.shape[1])
    if max_size> 1000:
        image = rescale(image, 1000./max_size)
        image = (image*255).astype(np.uint8)
    # use dlib to detect face
    # regress position map
    position_map = prn.process(image)
    return position_map

def get_300W_ground_truth_keypoints_and_vertices(ground_truth_folder, face_id, ground_truth_keypoint_indices, all_vertices_indices):
    mat_path = os.path.join(ground_truth_folder, face_id + '.mat')
    if not os.path.exists(mat_path):
        return None, None
    
    mat_file_content = loadmat(mat_path)
    vertices_ground_truth = np.array(mat_file_content['Fitted_Face']).T
    reduced_ground_truth_vertices = np.zeros((all_vertices_indices.shape[0],3))
    for i, v_index in enumerate(all_vertices_indices):
        reduced_ground_truth_vertices[i] = vertices_ground_truth[v_index]

    keypoints_ground_truth = np.zeros((68,3))
    for i, keypoint_index in enumerate(ground_truth_keypoint_indices):
        keypoints_ground_truth[i] = vertices_ground_truth[keypoint_index]
    return keypoints_ground_truth, vertices_ground_truth, reduced_ground_truth_vertices

#normalized mean squared error
def NME(vertices, ground_truth, normalization_factor):
    normalized_squared_error = np.zeros(vertices.shape[0])
    for i, vertex in enumerate(vertices):
        normalized_squared_error[i] = distance.euclidean(vertex,ground_truth[i]) / normalization_factor
    normalized_mean_squared_error = np.average(normalized_squared_error)
    return normalized_mean_squared_error

#normalized squared error
def normalized_squared_error(vertices, ground_truth, normalization_factor):
    normalized_squared_error = np.zeros(vertices.shape[0])
    for i, vertex in enumerate(vertices):
        normalized_squared_error[i] = distance.euclidean(vertex,ground_truth[i]) / normalization_factor
    return normalized_squared_error

if __name__ == '__main__':
    parser = ArgumentParser(description='PRNet2')

    parser.add_argument('--inputFolder', default='images', type=str,
                        help='path to folder containing input images')

    parser.add_argument('--groundTruthFolder', default='ground_truth', type=str,
                        help='path to folder containing ground truth vertices, assumes HELEN dataset file naming standard')
    
    parser.add_argument('--k', default=2, type=int,
                        help='k number of samples from dataset in inputFolder')

    parser.add_argument('--calculateKeypointsError', default= True, type=literal_eval,
                        help='should calculate kaypoints error, overrides calculateAllPointsError')

    parser.add_argument('--calculateAllPointsError', default= False, type=literal_eval,
                        help='should calculate error for all points, calculatKeypointsError overrides this')

    parser.add_argument('--sideAngle', default=80, type=int,
                        help='choose which angle the face of the side image should have')

    parser.add_argument('--saveOutput', default= False, type=literal_eval,
                        help='Save predicted vertices in .txt and average in .obj in the results folder both can be watched in MeshLab, overrides --calculateAllPointsError and --calculateKeypointsError')

    main(parser.parse_args())

#python PRNet2.py --inputFolder images --groundTruthFolder ground_truth --k 10 --calculateKeypointsError True --sideAngle 80
#python PRNet2.py --inputFolder ..\..\datasets\300W_LP\HELEN --groundTruthFolder ..\..\datasets\300W-3D-Face\HELEN --k 10 --calculateKeypointsError True --sideAngle 80