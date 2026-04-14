# coding=utf-8

# Standard Library Imports
import os

# Third Party Imports
import torch
import cv2
import random
import numpy as np
import networkx as nx

# Local Application/Library Specific Imports
from base.settings import LoggerMixin


class Aggregate(LoggerMixin):


    def __init__(self, phase_images, amp_images, img_path, results, image_ids):
        """
            Initializes the Aggregate object with provided images, image path, results, and image IDs.

            Parameters:
                images: List of image arrays to be processed.
                img_path: Path to the directory containing the images.
                results: List containing the results (predictions) from a model.
                image_ids: List containing the unique identifiers for each image in results.

            Attributes:
                device: The device (CPU/GPU) to be used for tensor computations. Default is 'cpu'.
                thresholds: Thresholds for different class pairs to decide if two nodes are connected in the graph.
                class_colors: Dictionary mapping class indices to corresponding colors for visualization.
                image_width, image_height: Dimensions of the input images.
                predictions: Filtered predictions for each image after processing.
                crop_image_ids, patch_ids, crop_images, patches, bboxes, class_label: Lists for storing processed data.
        """
        super().__init__()
        self.device = 'cpu' 
        self.thresholds = {(1, 1): 35, (1, 2): 23, (2, 2): 14.5}
        self.conf_thresholds = {0 : 0.5, 1 : 0.5, 2 : 0.5}
        self.class_colors = {0: (0, 0, 255), 1: (255, 0, 0), 2: (0, 255, 0), 3: (255, 0, 255)}
        self.image_width, self.image_height = 512, 384  
        self.phase_images = phase_images
        self.amp_images = amp_images
        self.img_path = img_path
        self.results = results
        self.image_ids = image_ids
        self.crop_image_ids, self.patch_ids, self.targeted_phase_images, self.targeted_amp_images, self.phase_patches, self.amp_patches, self.bboxes, self.class_label = [], [], [], [], [], [], [], []
        # Calculate and filter predictions for each image.
        self.predictions = self.calculate_all_predictions(results, image_ids)


    def calculate_all_predictions(self, results, image_ids, aspect_ratio_threshold=0.5, border_tolerance=10):
        """
        Filters bounding boxes in each image based on aspect ratio, border proximity, and class-specific confidence values.
        
        Parameters:
            results: List containing the results (predictions) from a model.
            image_ids: List containing the unique identifiers for each image in results.
            aspect_ratio_threshold: Threshold for the aspect ratio to filter bounding boxes.
            border_tolerance: Tolerance for filtering bounding boxes near the border of the image.
            
        Return:
            List of dictionaries containing filtered bounding boxes, class labels, and image_ids for each image.
        """
        filtered_results = []

        for result, img_id in zip(results, image_ids):
            if not result.boxes:
                # If no boxes are detected, append an empty entry to filtered_results.
                filtered_results.append({
                    'xywh': torch.tensor([]),
                    'cls': torch.tensor([]),
                    'conf': torch.tensor([]),
                    'image_id': img_id
                })
                continue

            bboxes_xywh = result.boxes.xywh  # Extract bounding boxes in xywh format.
            cls = result.boxes.cls  # Extract class labels.
            conf = result.boxes.conf

            # Extract coordinates from xyxy format for calculations.
            x1, y1, w, h = bboxes_xywh.t()
            x2 = x1 + w
            y2 = y1 + h

            width, height = w, h
            aspect_ratio = torch.max(width, height) / torch.min(width, height)  # Calculate aspect ratio for each box.

            # Determine whether each bounding box is close to the border of the image.
            on_left_border = x2 <= border_tolerance
            on_right_border = x1 >= 512 - border_tolerance
            on_top_border = y2 <= 25
            on_bottom_border = y1 >= 384 - border_tolerance

            # Initialize mask for filtering based on aspect ratio, border proximity, and confidence.
            to_remove = torch.zeros(len(bboxes_xywh), dtype=torch.bool)

            for i in range(len(bboxes_xywh)):
                class_id = int(cls[i].item())
                detection_threshold = self.conf_thresholds.get(class_id, 0)  # Default threshold is 0 if class_id is not in conf_thresholds
                # Define the conditions under which a bounding box should be removed.
                remove_condition = (
                                (conf[i] < detection_threshold) or \
                                (on_left_border[i] and y2[i] <= 384) or \
                                (on_right_border[i] and y1[i] <= 384) or \
                                (on_top_border[i] and x2[i] <= 512) or \
                                (on_bottom_border[i] and x1[i] <= 512)
                )
                if remove_condition:
                    to_remove[i] = True

            # Filter the bounding boxes and corresponding class labels based on the defined conditions.
            filtered_bboxes_xywh = bboxes_xywh[~to_remove]
            filtered_cls = cls[~to_remove]
            filtered_conf = conf[~to_remove]

            # Append filtered results for the current image to the filtered_results list.
            filtered_results.append({
                'xywh': filtered_bboxes_xywh if len(filtered_bboxes_xywh) > 0 else torch.tensor([]),
                'cls': filtered_cls if len(filtered_cls) > 0 else torch.tensor([]),
                'conf': filtered_conf if len(filtered_conf) > 0 else torch.tensor([]),
                'image_id': img_id
            })
        
        return filtered_results


    def compute_centroid(self, bbox):
        if bbox.nelement() == 0:  # Check if the bbox tensor is empty
            return torch.tensor([], device=self.device)

        x_center = bbox[:, 0]
        y_center = bbox[:, 1]
        centroids = torch.stack((x_center, y_center), dim=1)
        return centroids.to(self.device)


    def vectorized_distance_matrix(self, centroids, classes):

        # Create a mask of zeros with the same length as centroids.
        # centroids = torch.tensor(centroids, device=self.device)  # Convert to tensor if centroids are in numpy array
        centroids = centroids.clone().detach().to(self.device)
        mask = torch.zeros(len(centroids), dtype=torch.bool, device=self.device)
        valid_indices = torch.where(classes != 0)[0]
        mask[valid_indices] = True
        filtered_centroids = centroids[mask]

        # Now you can use unsqueeze
        A = filtered_centroids.unsqueeze(0)
        B = filtered_centroids.unsqueeze(1)
        distance_matrix = torch.sqrt(((A - B) ** 2).sum(-1))

        full_dists = torch.full((len(classes), len(classes)), float('inf'), device=centroids.device)
        valid_indices = torch.where(mask)[0]
        full_dists[valid_indices[:, None], valid_indices] = distance_matrix

        return full_dists


    def create_adjacency_graph_sparse(self, dist_mat, classes):

        G = nx.Graph()  # Initialize an empty undirected graph.
        class_list = classes.tolist()  # Convert classes tensor to a list for easier processing.
        non_zero_classes = {i for i, cls in enumerate(class_list) if cls != 0}  # Find indices of non-zero classes.
        G.add_nodes_from(non_zero_classes)  # Add nodes to the graph for each non-zero class.

        possible_class_pairs = set(self.thresholds.keys())  # Get the set of possible class pairs from thresholds.
        for i in non_zero_classes:
            for j in non_zero_classes:
                if i < j:  # Ensure each pair is processed only once
                    cls_i, cls_j = class_list[i], class_list[j]
                    if (cls_i, cls_j) in possible_class_pairs:
                        distance = dist_mat[i, j].item()
                        threshold = self.thresholds[(cls_i, cls_j)]
                        if distance < threshold:
                            G.add_edge(i, j)
                            # print(f"Adding edge between {i} and {j}: distance={distance}, threshold={threshold}")

        return G


    def find_aggregates(self, save_predicted_aggs: bool, save_predicted_wbc: bool, save_predicted_plt: bool, save_predicted_rbc: bool):
        """
            Identifies aggregates in each image, counts various cell types, and optionally saves annotated images.

            Parameters:
                save_predicted_aggs (bool): If True, saves images with detected aggregates.
                save_predicted_wbc (bool): If True, saves images with detected WBCs.
                save_predicted_plt (bool): If True, saves images with detected PLTs.
                save_predicted_rbc (bool): If True, saves images with detected RBCs.

            Returns:
                Tuple with aggregate information.
        """
        aggregates_list = []  # Initialize a list to hold information about aggregates in each image.
        aggregate_image_ids = []
        wbc_image_ids = []
        aggregate_image_info = []
        class_colors = {
        0: (0, 0, 255),   # For class 0, color Red
        1: (0, 255, 0),   # For class 1, color Green
        2: (255, 0, 0)    # For class 2, color Blue
            }
        for idx, prediction in enumerate(self.predictions):
            original_phase_img = self.phase_images[idx].copy()
            xywh = prediction['xywh'].to(self.device)  # Get the filtered bounding boxes for the current image.
            classes = prediction['cls'].to(self.device)  # Get the filtered class labels for the current image.
            confs = prediction['conf'].to(self.device)
            centroids = self.compute_centroid(xywh).to(self.device)  # Compute the centroids of the bounding boxes.
            # Compute the distance matrix and create an adjacency graph for the current image.
            dist_mat = self.vectorized_distance_matrix(centroids, classes)
            G = self.create_adjacency_graph_sparse(dist_mat, classes)
            # Find connected components in the graph, representing aggregates.
            aggregates = list(nx.connected_components(G))
            aggregates = [list(agg) for agg in aggregates if len(agg) > 1]  # Keep only aggregates with more than one node.
            # Initialize variables to hold various counts and lists related to aggregates.
            aggregate_class_counts = []
            counts_plt_plt, counts_plt_wbc, counts_wbc_wbc = 0, 0, 0
            plt_plt_centroids, plt_wbc_centroids, wbc_wbc_centroids = [], [], []
            counts_wbc = (classes == 1).sum().item()
            counts_plt = (classes == 2).sum().item()
            if len(aggregates) != 0:
                aggregate_image_ids.append(prediction.get('image_id'))
                bonding_box_img = original_phase_img.copy()
                # Iterate over detected cells to draw bounding boxes
                bboxes_xywh = xywh
                conf_count = 0
                for i, box in enumerate(bboxes_xywh):
                    x_center, y_center, w, h = box
                    # Convert center coordinates to corner coordinates.
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    
                    # color = class_colors.get(classes[i].item(), (0, 0, 0))  
                        
                    # bonding_box_img = cv2.rectangle(bonding_box_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    if classes[i].item() == 1 or classes[i].item() == 2:
                        # Get the color based on the cell class for this bounding box
                        color = class_colors.get(classes[i].item(), (0, 0, 0))  
                        
                        bonding_box_img = cv2.rectangle(bonding_box_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                        
                        # Add index_id for each cell on the image
                        index_id = f'{i+1}'  # Index starts from 1
                        # bonding_box_img = cv2.putText(bonding_box_img, index_id, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                        # Fetch the confidence value for this bounding box and display it.
                        conf = confs[i].item()
                        y_position = 15 + (conf_count * 15)
                        # bonding_box_img = cv2.putText(bonding_box_img, "{}: {:.2f}".format(index_id, conf), (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        conf_count += 1
                        
                # Display total WBC and PLT counts on the image
                # y_position = 15 + (conf_count * 15)  # Set the position below the last confidence value
                # bonding_box_img = cv2.putText(bonding_box_img, f"Total WBC: {counts_wbc}", (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

                # y_position += 15  # Move down for the next text
                # bonding_box_img = cv2.putText(bonding_box_img, f"Total PLT: {counts_plt}", (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        
                aggregate_color = (255, 0, 255)  # You can choose any color for the aggregate bounding box.
                for aggregate_indices in aggregates:
                    agg_bboxes = [xywh[i].cpu().numpy() for i in aggregate_indices]
                    x1s, y1s = [int(x - w / 2) for x, _, w, _ in agg_bboxes], [int(y - h / 2) for _, y, _, h in agg_bboxes]
                    x2s, y2s = [int(x + w / 2) for x, _, w, _ in agg_bboxes], [int(y + h / 2) for _, y, _, h in agg_bboxes]
                    min_x, min_y = min(x1s), min(y1s)
                    max_x, max_y = max(x2s), max(y2s)
                    aggregate_img = cv2.rectangle(bonding_box_img, (min_x, min_y), (max_x, max_y), aggregate_color, 1)
                    
                if save_predicted_aggs:
                    h, w, _ = original_phase_img.shape
                    final_img = np.zeros((h, 2*w, 3), dtype=np.uint8)
                    # Place the original image and the image with bounding boxes side by side on the new image
                    final_img[:, :w] = original_phase_img  # Original image
                    final_img[:, w:] = aggregate_img  # Image with bounding boxes + Aggregate Analysis
                    # final_img = aggregate_img
                    dir_parts = self.img_path.split("\\")
                    folder_name = "-".join(dir_parts[2: ]).lower()  # Getting the 'CFE001-0' and 'M1' parts and converting them to lowercase
                    new_img_name = f"{folder_name}-{prediction['image_id']}.png"
                    save_dir_base = self.config["img_output"]
                    save_dir = os.path.join(save_dir_base, folder_name, "aggs")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, new_img_name)
                    # cv2.imwrite(save_path, final_img)
                    
                    save_dir_phase = os.path.join(save_dir_base, folder_name, 'aggs', 'phase')
                    save_dir_predicted = os.path.join(save_dir_base, folder_name, 'aggs', 'predicted')
                    os.makedirs(save_dir_phase, exist_ok=True)
                    os.makedirs(save_dir_predicted, exist_ok=True)
                    save_path_phase = os.path.join(save_dir_phase, new_img_name)
                    save_path_predicted = os.path.join(save_dir_predicted, new_img_name)
                    cv2.imwrite(save_path_phase, original_phase_img)
                    cv2.imwrite(save_path_predicted, bonding_box_img)
                    
                    

            # if (classes == 1).sum().item() > 0 and save_predicted_wbc == True:
            if (classes == 1).sum().item() > 0:
                wbc_image_ids.append({prediction['image_id']})
                if save_predicted_wbc == True:
                        bonding_box_img = original_phase_img.copy()
                        bboxes_xywh = xywh
                        for i, box in enumerate(bboxes_xywh):
                            x_center, y_center, w, h = box
                            x1 = int(x_center - w / 2)
                            y1 = int(y_center - h / 2)
                            x2 = int(x_center + w / 2)
                            y2 = int(y_center + h / 2)
                            color = class_colors.get(classes[i].item(), (0, 0, 0))  
                            bonding_box_img = cv2.rectangle(bonding_box_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                        
                        h, w, _ = original_phase_img.shape
                        final_img = np.zeros((h, 2*w, 3), dtype=np.uint8)
                        final_img[:, :w] = original_phase_img
                        final_img[:, w:] = bonding_box_img

                        dir_parts = self.img_path.split("\\")
                        folder_name = "-".join(dir_parts[2: ]).lower()
                        # wbc_image_ids.append({prediction['image_id']})
                        new_img_name = f"{folder_name}-{prediction['image_id']}.png"
                        save_dir_base = self.config["img_output"]
                        save_dir = os.path.join(save_dir_base, folder_name, "wbc")
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, new_img_name)
                        # cv2.imwrite(save_path, final_img)
                        
                        save_dir_phase = os.path.join(save_dir_base, folder_name, 'wbc', 'phase')
                        save_dir_predicted = os.path.join(save_dir_base, folder_name, 'wbc', 'predicted')
                        os.makedirs(save_dir_phase, exist_ok=True)
                        os.makedirs(save_dir_predicted, exist_ok=True)
                        save_path_phase = os.path.join(save_dir_phase, new_img_name)
                        save_path_predicted = os.path.join(save_dir_predicted, new_img_name)
                        cv2.imwrite(save_path_phase, original_phase_img)
                        cv2.imwrite(save_path_predicted, bonding_box_img)
                        
                    
            if (classes == 2).sum().item() > 0 and save_predicted_plt == True:
                    bonding_box_img = original_phase_img.copy()
                    bboxes_xywh = xywh
                    for i, box in enumerate(bboxes_xywh):
                        x_center, y_center, w, h = box
                        x1 = int(x_center - w / 2)
                        y1 = int(y_center - h / 2)
                        x2 = int(x_center + w / 2)
                        y2 = int(y_center + h / 2)
                        color = class_colors.get(classes[i].item(), (0, 0, 0))  
                        bonding_box_img = cv2.rectangle(bonding_box_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                    
                    h, w, _ = original_phase_img.shape
                    final_img = np.zeros((h, 2*w, 3), dtype=np.uint8)
                    final_img[:, :w] = original_phase_img
                    final_img[:, w:] = bonding_box_img

                    dir_parts = self.img_path.split("\\")
                    folder_name = "-".join(dir_parts[2: ]).lower()
                    new_img_name = f"{folder_name}-{prediction['image_id']}.png"
                    save_dir_base = self.config["img_output"]
                    save_dir = os.path.join(save_dir_base, folder_name, "plt")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, new_img_name)
                    # cv2.imwrite(save_path, final_img)
                    
                    save_dir_phase = os.path.join(save_dir_base, folder_name, 'plt', 'phase')
                    save_dir_predicted = os.path.join(save_dir_base, folder_name, 'plt', 'predicted')
                    os.makedirs(save_dir_phase, exist_ok=True)
                    os.makedirs(save_dir_predicted, exist_ok=True)
                    save_path_phase = os.path.join(save_dir_phase, new_img_name)
                    save_path_predicted = os.path.join(save_dir_predicted, new_img_name)
                    cv2.imwrite(save_path_phase, original_phase_img)
                    cv2.imwrite(save_path_predicted, bonding_box_img)
                    

            if (classes == 0).sum().item() > 0 and save_predicted_rbc == True:
                    bonding_box_img = original_phase_img.copy()
                    bboxes_xywh = xywh
                    conf_count = 0
                    for i, box in enumerate(bboxes_xywh):
                        x_center, y_center, w, h = box
                        x1 = int(x_center - w / 2)
                        y1 = int(y_center - h / 2)
                        x2 = int(x_center + w / 2)
                        y2 = int(y_center + h / 2)
                        color = class_colors.get(classes[i].item(), (0, 0, 0))  
                        bonding_box_img = cv2.rectangle(bonding_box_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                        
                        index_id = f'{i+1}'  # Index starts from 1
                        bonding_box_img = cv2.putText(bonding_box_img, index_id, (int(x1), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

                        # Fetch the confidence value for this bounding box and display it.
                        conf = confs[i].item()
                        y_position = 15 + (conf_count * 15)
                        bonding_box_img = cv2.putText(bonding_box_img, "{}: {:.2f}".format(index_id, conf), (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                        conf_count += 1
                        
                    # Display total WBC and PLT counts on the image
                    y_position = 15 + (conf_count * 15)  # Set the position below the last confidence value
                    bonding_box_img = cv2.putText(bonding_box_img, f"Total WBC: {counts_wbc}", (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

                    y_position += 15  # Move down for the next text
                    bonding_box_img = cv2.putText(bonding_box_img, f"Total PLT: {counts_plt}", (5, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
                    
                    h, w, _ = original_phase_img.shape
                    final_img = np.zeros((h, 2*w, 3), dtype=np.uint8)
                    final_img[:, :w] = original_phase_img
                    final_img[:, w:] = bonding_box_img

                    dir_parts = self.img_path.split("\\")
                    folder_name = "-".join(dir_parts[2: ]).lower()
                    new_img_name = f"{folder_name}-{prediction['image_id']}.png"
                    save_dir_base = self.config["img_output"]
                    save_dir = os.path.join(save_dir_base, "rbc")
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, new_img_name)
                    cv2.imwrite(save_path, final_img)   

            agg_info_dict = {}
            # Iterate over detected aggregates to count different cell types and collect centroids.
            for aggregate in aggregates:

                cell_counts = {'r': 0, 'w': 0, 'p': 0}
                for index in aggregate:
                    cls = classes[index].item()
                    if cls == 0:
                        cell_counts['r'] += 1
                    elif cls == 1:
                        cell_counts['w'] += 1
                    elif cls == 2:
                        cell_counts['p'] += 1
                aggregate_class_counts.append(cell_counts)
                # aggregate_image_info.append((prediction.get('image_id'), f"Agg Counts: {len(aggregates)}", cell_counts)) 

                image_id = prediction.get('image_id')
                agg_count_info = f"Agg Counts: {len(aggregates)}"
                agg_type = ''

                if cell_counts['r'] == 0 and cell_counts['w'] > 1 and cell_counts['p'] == 0: 
                    agg_type = 'WBC-WBC Agg'
                    counts_wbc_wbc += 1
                    wbc_wbc_centroids.append(centroids[aggregate[0]])
                    # aggregate_image_info.append((prediction.get('image_id'), f"Agg Counts: {len(aggregates)}", 'WBC-WBC Agg', cell_counts))
                elif cell_counts['r'] == 0 and cell_counts['w'] > 0 and cell_counts['p'] > 0: 
                    agg_type = 'PLT-WBC Agg'
                    counts_plt_wbc += 1
                    plt_wbc_centroids.append(centroids[aggregate[0]])
                    # aggregate_image_info.append((prediction.get('image_id'), f"Agg Counts: {len(aggregates)}", 'PLT-WBC Agg', cell_counts))
                elif cell_counts['r'] == 0 and cell_counts['w'] == 0 and cell_counts['p'] > 1: 
                    agg_type = 'PLT-PLT Agg'
                    counts_plt_plt += 1
                    plt_plt_centroids.append(centroids[aggregate[0]])
                    # aggregate_image_info.append((prediction.get('image_id'), f"Agg Counts: {len(aggregates)}", "PLT-PLT Agg", cell_counts))
                if image_id not in agg_info_dict:
                    # agg_info_dict[image_id] = (image_id, agg_count_info, agg_type, cell_info)
                    agg_info_dict[image_id] = (image_id, agg_count_info, agg_type, cell_counts)
                else:
                    existing_info = agg_info_dict[image_id]
                    combined_info = existing_info + (agg_type, cell_counts)
                    agg_info_dict[image_id] = combined_info
            new_agg_info = list(agg_info_dict.values())
            aggregate_image_info.extend(new_agg_info)

            # Count the number of individual cells of each type in the current image.
            counts_rbc = (classes == 0).sum().item()
            counts_wbc = (classes == 1).sum().item()
            counts_plt = (classes == 2).sum().item()

            # Convert centroids to list for each class.
            rbc_centroids = centroids[classes == 0].tolist()
            wbc_centroids = centroids[classes == 1].tolist()
            plt_centroids = centroids[classes == 2].tolist()

            # Append a dictionary containing various information about the current image to the aggregates_list.
            aggregates_list.append({
                                    'image_id': prediction['image_id'],
                                    'aggregates': aggregates,
                                    'aggregate_class_counts': aggregate_class_counts,
                                    'counts_rbc': counts_rbc,
                                    'counts_wbc': counts_wbc,
                                    'counts_plt': counts_plt,
                                    'counts_plt_plt': counts_plt_plt,
                                    'counts_wbc_plt': counts_plt_wbc,
                                    'counts_wbc_wbc': counts_wbc_wbc,
                                    'plt_plt_centroids': plt_plt_centroids,
                                    'wbc_plt_centroids': plt_wbc_centroids,
                                    'wbc_wbc_centroids': wbc_wbc_centroids,
                                    'rbc_centroids': rbc_centroids,
                                    'wbc_centroids': wbc_centroids,
                                    'plt_centroids': plt_centroids
                                    })

        return aggregates_list, aggregate_image_ids, wbc_image_ids, aggregate_image_info


    def get_prediction_for_image_id(self, image_id):
        """
            Retrieves the filtered predictions for a given image_id.

            Parameters:
                image_id (str): The unique identifier of an image.

            Returns:
                dict or None: A dictionary containing the filtered bounding boxes, class labels,
                and image_id for the given image_id. Returns None if no matching prediction is found.
        """
        for prediction in self.predictions:
            if prediction['image_id'] == image_id:
                return prediction  # Return the prediction dictionary for the matching image_id.
        return None  # Return None if no matching prediction is found.


    def draw_objects(self, images, aggregates_list, draw_cells=True, draw_aggregates=True):
        """
            Visualizes bounding boxes and aggregates on input images.

            Parameters:
                images (list): A list of images on which to draw bounding boxes and aggregates.
                aggregates_list (list): A list of dictionaries containing information about aggregates in each image.
                draw_cells (bool, optional): Indicates whether to draw individual cell bounding boxes. Defaults to True.
                draw_aggregates (bool, optional): Indicates whether to draw bounding boxes around aggregates. Defaults to True.
        """
        for image, aggregate_info in zip(images, aggregates_list):
            image_id = aggregate_info["image_id"]  # Get the image_id for the current image.
            prediction = self.get_prediction_for_image_id(image_id)  # Get the filtered predictions for the current image.

            # Raise an exception if no prediction is found for the given image_id.
            assert prediction is not None, f"No prediction found for Image ID: {image_id}"

            # Retrieve various counts related to cells and aggregates in the current image.
            rbc_count = aggregate_info['counts_rbc']
            wbc_count = aggregate_info['counts_wbc']
            plt_count = aggregate_info['counts_plt']
            counts_plt_plt = aggregate_info['counts_plt_plt']
            counts_wbc_plt = aggregate_info['counts_wbc_plt']
            counts_wbc_wbc = aggregate_info['counts_wbc_wbc']

            # Print the counts for the current image.
            print(f"Image ID: {image_id.split('.')[0]}, RBC Count: {rbc_count}, WBC Count: {wbc_count}, PLT Count: {plt_count}, PLT-PLT Count: {counts_plt_plt}, WBC-PLT Count: {counts_wbc_plt}, WBC-WBC Count: {counts_wbc_wbc}")
            print("-" * 150)

            classes = prediction['cls']  # Get the filtered class labels for the current image.
            bboxes_xywh = prediction['xywh']  # Get the filtered bounding boxes for the current image.

            if draw_cells:
                # Draw individual cell bounding boxes and centroids on the image.
                for i in range(len(classes)):
                    x_center, y_center, w, h = bboxes_xywh[i]

                    # Convert center coordinates to corner coordinates.
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)

                    # Draw the bounding box and centroid on the image.
                    cv2.rectangle(image, (x1, y1), (x2, y2), self.class_colors[classes[i].item()], 2)
                    if draw_aggregates:
                        centroid = (x_center, y_center)
                        cv2.circle(image, (int(centroid[0]), int(centroid[1])), 3, self.class_colors[classes[i].item()], -1)

            if draw_aggregates:
                # Draw bounding boxes around aggregates on the image.
                for aggregate_indices in aggregate_info['aggregates']:
                    agg_bboxes = [bboxes_xywh[i] for i in aggregate_indices]
                    x1s, y1s = [int(x - w / 2) for x, _, w, _ in agg_bboxes], [int(y - h / 2) for _, y, _, h in agg_bboxes]
                    x2s, y2s = [int(x + w / 2) for x, _, w, _ in agg_bboxes], [int(y + h / 2) for _, y, _, h in agg_bboxes]
                    min_x, min_y = min(x1s), min(y1s)
                    max_x, max_y = max(x2s), max(y2s)
                    image = cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (255, 0, 255), 2)

            # Display the image with drawn bounding boxes and aggregates.
            if draw_cells or draw_aggregates:
                cv2.imshow('img', image)
                cv2.waitKey(0)


    def is_on_border(self, x1, y1, x2, y2):
        """
            Checks if coordinates (x1, y1) and (x2, y2) are on the predefined area's border.

            Parameters:
                x1, y1, x2, y2 (int): Coordinates of two points.

            Return:
                bool: True if on the border, False otherwise.
        """
        return (
            (x2 <= 512 and y2 <= 15) or 
            (x2 <= 15 and y2 <= 384) or  
            (369 <= y1 <= 384 and x1 <= 512) or 
            (497 <= x1 <= 512 and y1 <= 384)
        )


    def cut_patch(self, image: np.ndarray, bbox) -> np.ndarray:
        """
            Cuts an image patch from an input image according to a specified bounding box.
            
            If the bounding box lies completely or partly outside the image,
            these parts will be filled with np.nan for float arrays and -1 for integer arrays.

            Parameters:
                image (np.ndarray): The image from which the patch should be cut, in the format [w x h].
                bbox (list): Bounding box in the format [[row1, col1], [row2, col2]],
                            where "1" denotes the upper left corner and "2" denotes the lower right corner.
            Returns:
                np.ndarray: The cut patch from the input image.
        """
        height, width = image.shape
        row1, col1 = bbox[:2]
        row2, col2 = bbox[-2:]
        pad_top, pad_left, pad_bottom, pad_right = 0, 0, 0, 0

        # Check if top left corner exceeds the image dimensions
        if row1 < 0:
            pad_top = abs(row1)
            row1 = 0
        if col1 < 0:
            pad_left = abs(col1)
            col1 = 0
        # Check if bottom right corner exceed the image dimensions
        if height < row2:
            pad_bottom = abs(height - row2)
            row2 = height
        if width < col2:
            pad_right = abs(width - col2)
            col2 = width

        return np.pad(
            image[row1:row2, col1:col2],
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            "constant",
            constant_values=0,
        )


    def process_result(self, index, box, pred_class, img_array, amp_img_array, size):
        """
            Processes a single object detection result and returns relevant information.

            Parameters:
                index (int): Index of the current image.
                box (list): Bounding box of the detected object, in the format [x_center, y_center, width, height].
                pred_class (float): Predicted class of the detected object.
                img_array (np.ndarray): The image array where objects are detected.
                size (int): The size of the patch to be cut from the image.

            Returns:
                tuple: Contains image_id, bbox, img_patch, and label.
        """
        
        amp_min = amp_img_array.min()
        amp_max = amp_img_array.max()
        normalized_amp_img = 255 * (amp_img_array - amp_min) / (amp_max - amp_min)
        amp_img_array_int = normalized_amp_img.astype(np.uint8)
        # amp_img_array_int = normalized_amp_img.astype(np.uint8)
        celltype_mapping = {0.0: 'RBC', 1.0: 'WBC', 2.0: 'PLT'}
        label = celltype_mapping[pred_class]
        x_center, y_center, w, h = box
        # Convert center coordinates to corner coordinates.
        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox = [[x1, y1], [x2, y2]]
        y_center, x_center = (y1 + y2) // 2, (x1 + x2) // 2
        crop_bbox = [y_center - size // 2, x_center - size // 2, y_center + size // 2, x_center + size // 2]
        img_patch = self.cut_patch(img_array[:, :, 0], crop_bbox) / 255
        
        if len(self.amp_images) != 0:
            amp_img_patch = self.cut_patch(amp_img_array_int[:, :, 0], crop_bbox) / 255
            # amp_img_patch = img_patch
        else:
            amp_img_patch = img_patch
        return index, bbox, img_patch, amp_img_patch, label


    def crop_process(self):
        """
            Processes all predictions by cropping and processing image patches.

            This method iterates through all predictions, processes the detected bounding boxes, and prepares
            the relevant data for further analysis or storage.

            The internal states related to the images and image paths are reset to `None` after processing.

            Returns:
                tuple: Contains crop_image_ids, patch_ids, crop_images, patches, bboxes, and class_labels.
        """
        for idx, prediction in enumerate(self.predictions):
            self.process_boxes(idx, prediction)
        self.phase_images = None
        self.amp_images = None
        self.img_path = None
        return self.crop_image_ids, self.patch_ids, self.targeted_phase_images, self.targeted_amp_images, self.phase_patches, self.amp_patches, self.bboxes, self.class_label


    def process_boxes(self, idx, prediction):
        """
            Processes bounding boxes and classes for a given image, generating cropped patches.

            This method takes the predictions for an image and processes each bounding box,
            creating cropped patches if they belong to certain classes and optionally saving them.

            Parameters:
                idx (int): The index of the image in the list of images.
                prediction (dict): A dictionary containing bounding boxes, classes, and the image_id for the image.

            Note:
                The method updates the class attributes related to image_ids, patch_ids, cropped images,
                patches, bounding boxes, and class labels.
        """
        original_phase_img = self.phase_images[idx].copy()
        if len(self.amp_images) != 0:
            original_amp_img = self.amp_images[idx].copy()
            new_amp_img = np.repeat(original_amp_img[:, :, np.newaxis], 3, axis=2)
        else:
            original_amp_img = self.phase_images[idx].copy()
            new_amp_img = original_amp_img
        patch_id = 0
        bboxes_xywh = prediction['xywh'].to(self.device)  
        predicted_classes = prediction['cls'].to(self.device)   
        image_id = prediction['image_id']

        for box, pred_class in zip(bboxes_xywh, predicted_classes):

            if pred_class.item() in [1., 2.]:
                patch_id += 1
                image_id, bbox, phase_patch, amp_img_patch, label = self.process_result(idx, box, pred_class.item(), original_phase_img, new_amp_img, size=96)
                # if self.config["cropping_cells"]:
                    
                # Ensure the patch is in the right format
                phase_patch = (phase_patch * 255).astype(np.uint8)

                amp_patch = (amp_img_patch * 255).astype(np.uint8)

                if len(phase_patch.shape) == 3 and phase_patch.shape[2] == 1:
                    phase_patch = np.squeeze(phase_patch, axis=2)
                
                if self.config["phase_cropping"]:
                    dir_parts = self.img_path.split("\\")
                    folder_name = "-".join(dir_parts[3: ]).lower()  
                    # Create a new name for the image based on the directory and the image number
                    new_img_name = f"{folder_name}-{image_id}-phase_patch{patch_id}.png"
                    save_dir_base = self.config["img_output"]
                    save_phase_dir = os.path.join(save_dir_base, 'crop_phase')
                    os.makedirs(save_phase_dir, exist_ok=True)
                    save_path = os.path.join(save_phase_dir, new_img_name)
                    cv2.imwrite(save_path, phase_patch)
                
                if len(amp_patch.shape) == 3 and amp_patch.shape[2] == 1:
                    amp_patch = np.squeeze(amp_patch, axis=2)
                if self.config["amp_cropping"]:
                    new_phase_img_name = f"{folder_name}-{image_id}-amp_patch{patch_id}.png"
                    save_amp_dir = os.path.join(save_dir_base, 'crop_amp')
                    os.makedirs(save_amp_dir, exist_ok=True)
                    save_amp_path = os.path.join(save_amp_dir, new_phase_img_name)
                    cv2.imwrite(save_amp_path, amp_patch)
                    
                self.crop_image_ids.append(image_id)
                self.patch_ids.append(patch_id)
                self.targeted_phase_images.append(original_phase_img)
                self.targeted_amp_images.append(original_amp_img)
                self.phase_patches.append(phase_patch)
                self.amp_patches.append(amp_patch)
                self.bboxes.append(bbox)
                self.class_label.append(label)



