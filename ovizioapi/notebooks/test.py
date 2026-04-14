from ovizioapi.capture import OvizioCapture
import numpy as np
from ultralytics import YOLO
import torch
import numpy as np
import cv2
import timeit


capture_path ="C:/Users/Admin/Desktop/CFP007-0/M1/Capture 1.h5"
cpt = OvizioCapture(capture_path)
num_frames = len(cpt)
reconstructed_imgs = []
norm_image_list = []
output_dir = 'Z:/12_Clinical_Data/Ovizio_Output/test/'
cell_counts = {
        'RBC': 0,
        'WBC': 0,
        'PLT': 0
    }
print (f'Total Images: {num_frames}')
start_time = timeit.default_timer()
for i in range(num_frames):
    phase_image = cpt.get_phase(i)
    reconstructed_imgs.append(phase_image)
elapsed = timeit.default_timer() - start_time
print (f'Reconstruction Time: {elapsed}')

start_time = timeit.default_timer()
for i in range(len(reconstructed_imgs)):    
    new_phase_image = np.dstack([reconstructed_imgs[i], reconstructed_imgs[i], reconstructed_imgs[i]])
    norm_image = cv2.normalize(new_phase_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # norm_image = ((new_phase_image - np.min(new_phase_image)) * (1/(np.max(new_phase_image) - np.min(new_phase_image)) * 255)).astype('uint8')
    norm_image_list.append(norm_image)
elapsed = timeit.default_timer() - start_time
print (f'Normalize Time: {elapsed}')

start_time = timeit.default_timer()
detection_model = YOLO('C:/Users/Admin/Desktop/Projects/object_detect/v8xl_6_7_23_ovizio.pt')
batch_size = 10
for i in range(0, len(norm_image_list), batch_size):
    import pdb
    pdb.set_trace()
    batch_images = norm_image_list[i:i+batch_size]
    results = detection_model.predict(batch_images, classes=[0, 1, 2], 
                                        project=output_dir, name='', save=False, device='cuda:0')

    for result in results:
        boxes = result.boxes
        rbc_count = torch.sum(torch.eq(boxes.cls, 0.)).item()
        wbc_count = torch.sum(torch.eq(boxes.cls, 1.)).item()
        plt_count = torch.sum(torch.eq(boxes.cls, 2.)).item()
                
        cell_counts['RBC'] += rbc_count
        cell_counts['WBC'] += wbc_count  
        cell_counts['PLT'] += plt_count
elapsed = timeit.default_timer() - start_time
print (f'Detection Time: {elapsed}')

print(cell_counts)