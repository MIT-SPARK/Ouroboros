import cv2
import csv
import logging
from vlc_db.vlc_db import VlcDb
from vlc_db.spark_image import SparkImage
from tqdm import tqdm
from romatch import roma_outdoor
import torch
from PIL import Image
import numpy as np
from configs import supercloud_config as config

def read_timestamps(file_path):
    timestamps = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) != 2:
                continue
            timestamps.append((row[0], row[1].strip()))
    return timestamps

def read_image(idx, timestamps):
    timestamp, robot_name = timestamps[idx]
    image_path = f"../kimera_multi/outdoors/{robot_name}/images/{timestamp}.png"
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        logging.info(f"Not Found: {image_path}")
    return image

def main():
    logging.basicConfig(level=logging.INFO)
    
    output_file_path = "../kimera_multi/outdoors/output.csv"
    timestamps_file_path = "../kimera_multi/outdoors/timestamps.csv"

    downsampled_shape = (308, 238)
    
    timestamps = read_timestamps(timestamps_file_path)
    lcd = VlcDb(config)
    # weights = torch.load('RoMa.pth')
    # dinov2_weights = torch.load('DINOv2.pth')
    # roma_model = roma_outdoor(weights=weights, dinov2_weights=dinov2_weights, coarse_res=downsampled_shape, upsample_res=downsampled_shape, device='cuda')
    
    with open(output_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        
        for idx, (timestamp, robot_name) in tqdm(enumerate(timestamps)):

            print(timestamp,robot_name)
            
            image = read_image(idx, timestamps)
            rgb_img = cv2.resize(image, (308, 238))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_img = rgb_img.astype(np.float32)
            spark_image = SparkImage(rgb_img,None)
            embedding = lcd.get_embedding(spark_image)
            best_frame, IP = lcd.query_embeddings(embedding,2)
            uuid = lcd.add_image(robot_name,timestamp,spark_image)
            lcd.update_embedding(uuid,embedding)
            print(best_frame,IP)
            
            # best_frame_image = read_image(best_frame, timestamps)
            # warp, certainty = roma_model.match(Image.fromarray(cv2.resize(image, downsampled_shape)),Image.fromarray(cv2.resize(best_frame_image, downsampled_shape)), device='cuda')
            # matches, certainty = roma_model.sample(warp, certainty)
            # kpts1, kpts2 = roma_model.to_pixel_coordinates(matches, *downsampled_shape, *downsampled_shape)
            # concatenated_image = cv2.vconcat([image, best_frame_image])
            # cv2.imshow(robot_name, concatenated_image)
            # cv2.waitKey(1)
            
            # writer.writerow([idx, best_frame, IP])

if __name__ == "__main__":
    main()