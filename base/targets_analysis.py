import numpy as np
import os
from morphology import *
import time
from reconstructor import Reconstructor
import h5py
import timeit
from ovizioapi.capture import OvizioCapture
from ovizioapi import OvizioApiNet
import cv2
import sys
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision.utils import save_image


holo_images, reconstruct_imgs, phase_images, norm_image_list = [], [], [], []



def h5_length(h5_file):
    with h5py.File(h5_file, 'r') as f:
        return len(f)


def prepare_dataset(h5_file):
    # global holo_images
    time_step = 0
    with h5py.File(h5_file, 'r') as f:
        while True:
            path = f'/timeStep{time_step}/hologram/data'
            if path in f:
                ds = f[path]
                if isinstance(ds, h5py.Dataset):
                    holo_images.append(np.array(ds).reshape(1536, 2048))
                time_step += 1
            else:
                break

    return holo_images


def get_images_slice(start_idx: int, end_idx: int):
    # global holo_images
    if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
        return ValueError('Invalid Data.')
    elif len(holo_images) == 0:
        return ValueError('There is no data.')
    elif end_idx > len(holo_images):
        end_idx = len(holo_images)
    return holo_images[start_idx:end_idx]


def background_subtraction(input_images):
    background = np.mean(input_images[:100], axis=0)
    norm_background = cv2.normalize(background, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return norm_background


def analyze_container(input_path, output_path, batch, experiment_name):
    path_len = h5_length(input_path)
    div = path_len // batch
    idx1 = 0

    print('The total number of images is ', path_len)
    print('The total number of batches is ', div+1)
    print('/n')

    time1 = time.time()

    for i in range(0, div+1):

        time_loop1 = time.time()

        idx2 = i*batch + batch - 1
        if idx2 > path_len:
            idx2 = path_len

        print(f"{i+1}/{div+1}: {idx1} - {idx2}")

        with h5py.File(input_path,'r') as f:
            phases=f['images']['patches'][()]
            
        img_index = idx1
        for phase in phases:
            contours, norm_phase = im2contour(phase)
            
            for i in range(len(contours)):
                destination = os.path.join(output_path + "/", experiment_name)
                img_thresholding(phase, norm_phase, contours, i, img_index, destination, experiment_name, save_csv=True)
        
            img_index += 1
        idx1 = idx2 + 1
        time_loop2 = time.time()
        print('The loop time is ', time_loop2-time_loop1)
        print('/n')

    time2 = time.time()
    print('The execution time is ', time2-time1)
    
    return


def analyze_qc(input_path, csv_path, output_directory, save_images: bool):
    feature_list = []
    reconstruct = Reconstructor(device='cuda',
                        fin_net_path = 'C:/Users/Admin/Desktop/Projects/object_detect/fin_pwa_32-64-64-32_rep_torchv2.pth',
                        cnn_net_path = 'C:/Users/Admin/Desktop/Projects/object_detect/cnn_pwa_32-64-64-32_rep_torchv2.pth')
        
    holo_images = prepare_dataset(input_path)

    print(f'Total Images to reconstruct: {len(holo_images)}')

    chunk_size = 2 # Number of images to reconstruct
    num_chunks = len(holo_images) // chunk_size
    # Check if num_chunks is zero and raise an error if it is
    if num_chunks == 0:
        raise ValueError("Number of chunks is zero, please ensure there are enough images to process.")

    for chunk in range(num_chunks + 1):
        frame_id1 = chunk * chunk_size
        frame_id2 = frame_id1 + chunk_size
        holo = get_images_slice(frame_id1, frame_id2)
        # print(f'Reconstractig Images {frame_id1} to {frame_id2}')
        holo = np.stack(holo)[None, :]      # convert a list of 2d array into 4d array, shape [1, len, h, w], 0-255
        start_time = timeit.default_timer()
        phase, amp = reconstruct.reconstruct(holo)
        # print(f'{timeit.default_timer() - start_time, holo.shape}, {phase.shape}, {amp.shape}')
        for i in range(len(phase)):
            if save_images:
                output_file_path = os.path.join(output_directory, f'phase_img{frame_id1 + i}.png')
                save_image(phase[i], output_file_path)
            reconstructed_data = phase[i].detach().cpu().numpy()
            # reconstructed_data = phase[i]
            reconstruct_imgs.append(reconstructed_data)
            
    background = background_subtraction(reconstruct_imgs)
    for i,phase in enumerate(reconstruct_imgs):
        contours, norm_phase = im2contour(phase, background)
        for i in range(len(contours)):
            result = qc_img_thresholding(phase, contours, i)
            if result:
                feature_list.append(result)
                
    # Given column names
    columns = ['Equivalent Diameter', 'Aspect Ratio', 'Area', 'Physical Volume', 'Perimeter', 'Circularity', 'Sphericity', 'Phase shift', 'Optical height', 'Physical height', 'Optical Volume/nFrame']

    # Convert feature_list to a DataFrame
    df = pd.DataFrame(feature_list, columns=columns)

    return df


def reconstruct_images(save_images: bool, ovizio_reconstruction: bool, h5_file_path: str):
        
    if ovizio_reconstruction:
        h5_file_path = h5_file_path
        np.set_printoptions(threshold=sys.maxsize)
        OvizioApiNet.OvizioApiNet.set_UseGPU(True)
        cpt = OvizioCapture(h5_file_path)
        num_frames = len(cpt)
        num_frames = (len(cpt) // 130) * 130
        # num_frames = 600
        for i in range(num_frames):
            phase_image = cpt.get_phase(i)
            phase_images.append(phase_image)
        print(f'Total Reconstructed Images: {num_frames}')
        # Normalization Process
        # for i in range(len(phase_images)): 
        #     new_phase_image = np.dstack([phase_images[i], phase_images[i], phase_images[i]])
        #     norm_image = cv2.normalize(new_phase_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        #     norm_image_list.append(norm_image)
    else:
        pass

    return phase_images


def object_detection_batch(total_images, model, batch_size, device):
        
    batch_image_ids, batch_patch_ids, batch_images, batch_patches, batch_masked_patches, batch_bboxes, batch_cls = [], [], [], [], [], [], []
    RBC, WBC, PLT, patch_id, total_cell = 0, 0, 0, -1, 0
    
    color_dict = {
    'RBC': (0, 0, 255),  # Red color for RBC
    'WBC': (0, 255, 0),  # Green color for WBC
    'PLT': (255, 0, 0)   # Blue color for PLT
    }
    
    class_names = {0: 'RBC', 1: 'WBC', 2: 'PLT'}
    # total_images_list = sorted(glob.glob(total_images), key=len)
    total_images_list = total_images
    for i in range(0, len(total_images_list), batch_size):
        # loop for total_images / batch_size times
        batch_imgs = total_images_list[i:i+batch_size]
        normalize_images = []
        for j in range(len(batch_imgs)):
            new_phase_image = np.dstack([batch_imgs[j], batch_imgs[j], batch_imgs[j]])
            normalize_img = cv2.normalize(new_phase_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # normalize_img = np.array(cv2.imread(batch_imgs[j]))
            normalize_images.append(normalize_img)
        try:
            results = model.predict(normalize_images, classes=[0, 1, 2], name='', save=False, device=device, show_labels=False, save_crop=False, verbose=False, imgsz=(384,512))

            # process each image
            for k, result in enumerate(results):
                boxes = result.boxes
                rbc_count, wbc_count, plt_count = 0, 0, 0
                # process each cell
                pred_classes = result.boxes.cls.cpu().data.numpy()
                for box, pred_class in zip(boxes, pred_classes):
                    # filter cells on borders
                    x1, y1, x2, y2 = box.xyxy[0]
                    width, heith = x2-x1, y2-y1
                    ratio = max(width, heith) / min(width, heith)

                    if ratio > 2 and (
                            (x2 <= 512 and y2 <= 15) or 
                            (x2 <= 15 and y2 <= 384) or 
                            (369 <= y1 <= 384 and x1 <= 512) or 
                            (497 <= x1 <= 512 and y1 <= 384)
                        ):
                        # self.logger.info(f'No. {i+k} image has cropped cell(s) on the border')
                        pass
                    else:
                        total_cell += 1
                        if pred_class == 0:
                            RBC += 1
                            rbc_count += 1
                    
                        if pred_class == 1 or pred_class == 2:  # detect WBC or PLT
                            patch_id += 1

                            if pred_class == 1:
                                WBC += 1
                                wbc_count += 1
                            else:
                                PLT += 1
                                plt_count += 1

        except Exception as e:
            torch.cuda.empty_cache()
            print(f"An error occurred: {e}")

    # df_batch_counts = pd.DataFrame([[total_cell, RBC, WBC, PLT]], columns=['n_cells', 'n_rbcs', 'n_wbcs', 'n_plts'])
    df_batch_counts = pd.DataFrame([[total_cell, RBC, WBC, PLT]], columns=['n_cells', 'n_rbcs', 'n_wbcs', 'n_plts'])

    torch.cuda.empty_cache()
    return  df_batch_counts

def object_detection_batch5000(total_images, model, batch_size, device):
    batch_image_ids, batch_patch_ids, batch_images, batch_patches, batch_masked_patches, batch_bboxes, batch_cls = [], [], [], [], [], [], []
    RBC, WBC, PLT, patch_id, total_cell = 0, 0, 0, -1, 0
    
    color_dict = {
        'RBC': (0, 0, 255),  # Red color for RBC
        'WBC': (0, 255, 0),  # Green color for WBC
        'PLT': (255, 0, 0)   # Blue color for PLT
    }
    
    class_names = {0: 'RBC', 1: 'WBC', 2: 'PLT'}
    
    total_images_list = total_images
    
    df_list = []  # Step 1: Initialize a list to hold your data frames
    image_counter = 0  # Counter to track the number of processed images
    
    results = None
    
    for i in range(0, len(total_images_list), batch_size):
        batch_imgs = total_images_list[i:i+batch_size]
        normalize_images = []
        for j in range(len(batch_imgs)):
            new_phase_image = np.dstack([batch_imgs[j], batch_imgs[j], batch_imgs[j]])
            normalize_img = cv2.normalize(new_phase_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            normalize_images.append(normalize_img)
        try:
            results = model.predict(normalize_images, classes=[0, 1, 2], name='', save=False, device=device, show_labels=False, save_crop=False, verbose=False, imgsz=(384,512))

            for k, result in enumerate(results):
                # start
                boxes = result.boxes
                rbc_count, wbc_count, plt_count = 0, 0, 0
                pred_classes = result.boxes.cls.cpu().data.numpy()
                for box, pred_class in zip(boxes, pred_classes):
                    x1, y1, x2, y2 = box.xyxy[0]
                    width, heith = x2-x1, y2-y1
                    ratio = max(width, heith) / min(width, heith)
                # end
                    if ratio > 2 and (
                            (x2 <= 512 and y2 <= 15) or 
                            (x2 <= 15 and y2 <= 384) or 
                            (369 <= y1 <= 384 and x1 <= 512) or 
                            (497 <= x1 <= 512 and y1 <= 384)
                        ):
                        # self.logger.info(f'No. {i+k} image has cropped cell(s) on the border')
                        pass
                    else:
                        total_cell += 1
                        if pred_class == 0:
                            RBC += 1
                            rbc_count += 1

                        if pred_class == 1 or pred_class == 2:
                            patch_id += 1

                            if pred_class == 1:
                                WBC += 1
                                wbc_count += 1
                            else:
                                PLT += 1
                                plt_count += 1

        except Exception as e:
            results = None
            torch.cuda.empty_cache()
            print(f"An error occurred: {e}")

        image_counter += len(batch_imgs)
        counter_val = 5000
        if image_counter >= counter_val:
            df_batch_counts = pd.DataFrame([[total_cell, RBC, WBC, PLT]], columns=['n_cells', 'n_rbcs', 'n_wbcs', 'n_plts'])
            df_list.append(df_batch_counts)
            total_cell, RBC, WBC, PLT = 0, 0, 0, 0
            image_counter = 0
            print('image_counter: ', image_counter)

    if image_counter > 0:
        df_batch_counts = pd.DataFrame([[total_cell, RBC, WBC, PLT]], columns=['n_cells', 'n_rbcs', 'n_wbcs', 'n_plts'])
        df_list.append(df_batch_counts)

    result_df = pd.concat(df_list, ignore_index=True)
    print(f"Each Batch count {counter_val}")
    results = None
    total_images_list = None
    torch.cuda.empty_cache()
    return result_df


def analyze_ovizioapi_qc (input_path, csv_path, output_directory, save_images: bool):
    feature_list = []
    reconstruct_imgs = reconstruct_images(save_images=False, ovizio_reconstruction=True, h5_file_path=input_path)
            
    background = background_subtraction(reconstruct_imgs)
    for i,phase in enumerate(reconstruct_imgs):
        contours, norm_phase = im2contour(i, phase, background)
        for i in range(len(contours)):
            result = qc_img_thresholding(phase, contours, i)
            if result:
                feature_list.append(result)
                
                
    # Given column names
    columns = ['Equivalent Diameter', 'Aspect Ratio', 'Area', 'Physical Volume', 'Perimeter', 'Circularity', 'Sphericity', 'Phase shift', 'Optical height', 'Physical height', 'Optical Volume/nFrame']

    # Convert feature_list to a DataFrame
    df = pd.DataFrame(feature_list, columns=columns)

    return df


def KD_method():
    path  = input("Enter the path of the h5 file: ").replace('"','')
    output_path = input("Enter the path of the output folder: ").replace('"','')
    experiment_name = input("Enter the name of the experiment: ").replace('"','')
    print('/n')

    destination = os.path.join(output_path + "/", experiment_name)
    if not os.path.exists(destination):
        os.makedirs(destination)

    with open(f'{destination}/{experiment_name}.csv', 'a') as f:
        header = ['Frame', 'Cell Index', 'Equivalent Diameter', 'Aspect Ratio', 'Area', 'Physical Volume', 'Perimeter', 'Circularity', 'Sphericity', 'Phase shift', 'Optical height', 'Physical height', 'Optical Volume']
        f.write(','.join(header) + '/n')

    reconstruct = Reconstructor(device='cuda',
                                fin_net_path='C:/Users/Admin/Desktop/Projects/dhm/models/fin_pwa_32-64-64-32_rep_torchv2.pth',
                            cnn_net_path='C:/Users/Admin/Desktop/Projects/dhm/models/cnn_pwa_32-64-64-32_rep_torchv2.pth')

    path_len = h5_length(path)
    batch = 130
    div = path_len // batch
    idx1 = 0

    print('The total number of images is ', path_len)
    print('The total number of batches is ', div+1)
    print('/n')

    time1 = time.time()

    for i in range(0, div+1):

        time_loop1 = time.time()

        idx2 = i*batch + batch - 1
        if idx2 > path_len:
            idx2 = path_len

        print(f"{i+1}/{div+1}: {idx1} - {idx2}")

        hologram = prepare_dataset(path, idx1, idx2)
        dataset = np.stack(hologram)[None, :]
        phases, _ = reconstruct.reconstruct(dataset)
            
        img_index = idx1
        for phase in phases:
            phase = phase.cpu().numpy()
            contours, norm_phase = im2contour(phase)
            
            for i in range(len(contours)):        
                img_thresholding(phase, norm_phase, contours, i, img_index, destination, experiment_name)
        
            img_index += 1
        idx1 = idx2 + 1
        time_loop2 = time.time()
        print('The loop time is ', time_loop2-time_loop1)
        print('/n')

    time2 = time.time()
    print('The execution time is ', time2-time1)