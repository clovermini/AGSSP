import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    # Convert YOLO boxes (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    # Return IoU
    return inter_area / union_area if union_area > 0 else 0


def non_max_suppression(yolo_boxes, iou_threshold=0.8):
    """Perform Non-Maximum Suppression (NMS) on YOLO boxes."""
    # Sort boxes by confidence score (descending)
    yolo_boxes = sorted(yolo_boxes, key=lambda x: x[0], reverse=True)

    nms_boxes = []
    while yolo_boxes:
        # Select the box with the highest confidence
        chosen_box = yolo_boxes.pop(0)
        nms_boxes.append(chosen_box)

        # Compare IoU with remaining boxes
        yolo_boxes = [
            box
            for box in yolo_boxes
            if iou(chosen_box, box) < iou_threshold
        ]

    return nms_boxes


def process_and_visualize_yolo(anomaly_map_path, image_path, yolo_output_path, threshold=0.5, top_k=5, class_id=0, save_mask=False, save_mask_path=None):
    """
    Process an anomaly map, resize it to match the image size, generate YOLO labels,
    and visualize anomaly map, binary map, and YOLO bounding boxes.

    Parameters:
        anomaly_map_path (str): Path to the .npy anomaly map file.
        image_path (str): Path to the corresponding image for resizing.
        yolo_output_path (str): Path to save YOLO format annotations.
        threshold (float): Threshold to binarize the anomaly map.
        min_area (int): Minimum size of connected regions to keep.
        class_id (int): Class ID for YOLO labels.
    """
    # Load the anomaly map
    anomaly_map = np.load(anomaly_map_path)
    #print('anomaly_map max ', np.max(anomaly_map), ' -- ', np.min(anomaly_map))
    
    # Load the reference image to get the size
    image = cv2.imread(image_path)
    image_h, image_w = image.shape[:2]
    
    # Resize the anomaly map to match the image size
    anomaly_map_resized = cv2.resize(anomaly_map, (image_w, image_h))
    
    # Binarize the anomaly map using the threshold
    binary_map = (anomaly_map_resized > threshold).astype(np.uint8) * 255  # Convert to 0 and 255
    if save_mask:
        #print('anomaly_map_path ', anomaly_map_path)
        #print('image_path ', image_path)
        #print('binary_map ', binary_map.shape, ' image ', image.shape)
        cv2.imwrite(save_mask_path, binary_map)
        return 1

    min_area = max((image_h//100)*(image_w//100), 10)
    # Find contours (connected components)
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Prepare YOLO labels
    yolo_labels = []
    binary_map_with_boxes = image  #cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)

    boxes = []
    for contour in contours:

        # Calculate the area of the contour
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # Skip small regions

        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Calculate region probability
        region = anomaly_map_resized[y:y+h, x:x+w]
        prob = np.max(region)

        # Convert to YOLO format
        x_center = (x + w / 2) / anomaly_map_resized.shape[1]
        y_center = (y + h / 2) / anomaly_map_resized.shape[0]
        width = w / anomaly_map_resized.shape[1]
        height = h / anomaly_map_resized.shape[0]

        # Append box
        boxes.append((prob, class_id, x_center, y_center, width, height))  # Class ID 0 for anomalies

    # Step 3: Sort boxes by probability and take top_k
    boxes = sorted(boxes, key=lambda x: x[0], reverse=True)[:top_k]

    # Step 4: Apply NMS
    nms_boxes = non_max_suppression(boxes, iou_threshold=0.8)
    print('nms_boxes ', len(nms_boxes))
    
    for box in nms_boxes:
        prob, class_id, x_center, y_center, width, height = box
        h, w = image.shape[:2]
        # Convert YOLO format to pixel coordinates
        x_min = int((x_center - width / 2) * w)
        y_min = int((y_center - height / 2) * h)
        x_max = int((x_center + width / 2) * w)
        y_max = int((y_center + height / 2) * h)
        
        # Draw rectangle
        cv2.rectangle(binary_map_with_boxes, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Save YOLO labels to file
    with open(yolo_output_path, 'w') as f:
        for label in nms_boxes:
            f.write(f"{label[1]} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f} {label[5]:.6f}\n")
    
    return 1


# Example usage (replace paths with actual files)

cnt = 0

anomaly_dir = '/home/data/Datasets/public/tmp_features_huge/casting_billet/images/'  # max_mean  0.987919  change to your path
#anomaly_dir = '/home/data/Datasets/public/tmp_features_huge/pipeData/images/'  # max_mean  0.92978936  change to your path

anomaly_list = os.listdir(anomaly_dir)
anomaly_map_max_list = []

# get mean_max
for item in anomaly_list:
    if not item.endswith('.npy'):
        continue

    anomaly_map_path = anomaly_dir + item

    image_path = anomaly_map_path.replace('/home/data/Datasets/public/tmp_features_huge/', '/home/data/Datasets/public/').replace('_anomaly.npy', '.jpg')
    if not os.path.exists(image_path) or not os.path.join(anomaly_map_path):
        print('not exists! ', anomaly_map_path, ' -- ', image_path)
        continue
    cnt += 1

    anomaly_map = np.load(anomaly_map_path)
    anomaly_map_max_list.append(np.max(anomaly_map))

max_mean = np.mean(anomaly_map_max_list)
print('max_mean ', max_mean)

# generate mask and yolo
for item in anomaly_list:
    if not item.endswith('.npy'):
        continue
    anomaly_map_path = anomaly_dir + item
    image_path = anomaly_map_path.replace('/home/data/Datasets/public/tmp_features_huge/', '/home/data/Datasets/public/').replace('_anomaly.npy', '.jpg')


    process_and_visualize_yolo(
        anomaly_map_path=anomaly_map_path,
        image_path=image_path,
        yolo_output_path=anomaly_map_path.replace('.npy', '.txt'),
        threshold=(max_mean-0.1),   # Adaptive adjustment based on the dataset
        top_k=10,
        class_id=0,
        save_mask=False, 
        save_mask_path=anomaly_map_path.replace('.npy', '_mean_max_0.1.png'),  # '_mean_max_0.35.png'  _0.5.png
    )
        
    if cnt % 1000 == 0:
        print('handle ... ', cnt, ' max_mean ', max_mean)

print('done! ', cnt)


