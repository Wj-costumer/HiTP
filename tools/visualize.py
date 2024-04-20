# visualize.py
import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

__all__ = ["visualize_camera", "visualize_lidar", "visualize_map"]

NameMapping = {
    'movable_object.barrier': 'barrier', 
    'vehicle.bicycle': 'bicycle', 
    'vehicle.bus.bendy': 'bus', 
    'vehicle.bus.rigid': 'bus', 
    'vehicle.car': 'car', 
    'vehicle.construction': 'construction_vehicle', 
    'vehicle.motorcycle': 'motorcycle', 
    'human.pedestrian.adult': 'pedestrian', 
    'human.pedestrian.child': 'pedestrian', 
    'human.pedestrian.construction_worker': 'pedestrian', 
    'human.pedestrian.police_officer': 'pedestrian', 
    'movable_object.trafficcone': 'traffic_cone', 
    'vehicle.trailer': 'trailer', 
    'vehicle.truck': 'truck'
    }

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
    "tricycle": (220, 20, 60),  # 相比原版 mmdet3d 的 visualize 增加 tricycle
    "cyclist": (220, 20, 60)  # 相比原版 mmdet3d 的 visualize 增加 cyclist
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

IMGS_PATH = "./projects/UniAD/data/nuscenes"
OUTPUT_PATH = "./outputs"
pc_range = 100
image_size = 1024

def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == np.bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

    
def combine_all(images, combine_save_path):
    """将 6 个视角的图片和 bev视角下的 lidar 进行拼接

    :param img_path_dict: 每个视角的 img 图片路径 && bev 视角下 lidar 的图片路径

    |----------------------------------------------  | -------------
    |cam_front_left | cam_front  | cam_front_right   |
    |-------------- | ---------  | ---------------   |  lidar_bev
    |cam_back_left  | cam_back   | cam_back_right    |  
    |----------------------------------------------  | -------------

    """
    cam_front = images["CAM_FRONT"]
    cam_front_left = images["CAM_FRONT_LEFT"]
    cam_front_right = images["CAM_FRONT_RIGHT"]
    cam_back = images["CAM_BACK"]
    cam_back_left = images["CAM_BACK_LEFT"]
    cam_back_right = images["CAM_BACK_RIGHT"]
    # merge img
    front_combined = cv2.hconcat([cam_front_left, cam_front, cam_front_right])
    back_combined = cv2.hconcat([cam_back_right, cam_back, cam_back_left])
    back_combined = cv2.flip(back_combined, 1)  # 左右翻转
    img_combined = cv2.vconcat([front_combined, back_combined])
    # 读 lidar
    lidar_bev = images["lidar"]
    # img_combined 等比例缩小
    target_height = lidar_bev.shape[0]
    scale_factor = target_height / img_combined.shape[0]
    target_width = int(img_combined.shape[1] * scale_factor)
    img_combined = cv2.resize(img_combined, (target_width, target_height))
    # merge all
    merge_image = cv2.hconcat([img_combined, lidar_bev])
    # 保存图片
    cv2.imwrite(combine_save_path, merge_image)
    
    return merge_image

def get_matrix(calibrated_data, inverse = False): 
    output = np.eye(4)
    output[:3, :3] = Quaternion(calibrated_data["rotation"]).rotation_matrix
    output[:3, 3] = calibrated_data["translation"]
    if inverse:
        output = np.linalg.inv(output)
    return output

def visualize_lidar_points(
    lidar_file_path: str,
    ego2lidar,
    pc_range: int,
    image_size: int
) -> np.array:
    mmcv.check_file_exist(lidar_file_path)
    points = np.frombuffer(open(lidar_file_path, 'rb').read(), dtype=np.float32)
    points = points.reshape(-1, 5)[:, :3] # x, y, z, intensity, ring_idx
    points = np.concatenate([points, np.ones((len(points), 1))], axis = 1)
    points = points @ ego2lidar.T # convert to lidar
    pc_range = 100
    points[:, :2] /= pc_range
    image_size = 1024
    # 将pts缩放成图像大小，并平移到中心
    points[:, :2] = points[:, :2] * image_size / 2 + image_size / 2
    
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for ix, iy, iz in points[:, :3]:
        if 0 <= ix < image_size and 0 <= iy <image_size:
            image[int(ix), int(iy)] = 255, 255, 255 
    return image

def visualize_2D_boxes(
    nusc,
    image, 
    boxes,
    pc_range: int,
    image_size: int 
    ) -> np.array:
    # vsiualize 2D boxes
    for box in boxes:
        x, y, z = box.center
        w, h, l = box.wlh
        corners = [(x + w/2, y + h/2), (x + w/2, y - h/2), (x - w/2, y - h/2), (x - w/2, y + h/2)]
        corners = [(int(x / pc_range * image_size / 2 + image_size / 2), int(y / pc_range * image_size / 2 + image_size / 2)) for x, y in corners]
        annotation = nusc.get('sample_annotation', box.token)
        if box.name in NameMapping:
            name = NameMapping[box.name]
            color = OBJECT_PALETTE[name]
            for i in range(4):
                cv2.line(image, corners[i], corners[(i + 1) % 4], color, 2)
    return image

def visualize_3D_boxes(
    nusc,
    image_path,
    boxes,
    camera_intrinsic,
    pc_range,
    image_size
) -> np.array:
    
    image_cam = cv2.imread(image_path)
    for box in boxes:
        corners = box.corners().T # 8 x 3
        corners = np.concatenate([corners, np.ones((len(corners), 1))], axis = 1) @ camera_intrinsic.T # 8 x 4
        corners[:, :2] /= corners[:, [2]]
        if box.name in NameMapping:
            name = NameMapping[box.name]
            color = OBJECT_PALETTE[name]
            for start, end in [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)]:
                cv2.line(image_cam, tuple(corners[start][:2].astype(np.int)), tuple(corners[end][:2].astype(np.int)), color, thickness=2)
    return image_cam

def image_to_video(file_path, output):
    img_list = os.listdir(file_path)  # 生成图片目录下以图片名字为内容的列表
    img_list = img_list.sort(key = lambda x: int(x.split('.png')[0][-1])) 
    print("----", len(img_list))
    test_frame = cv2.imread(file_path + '/' + img_list[0])
    weight, height, channel = test_frame.shape
    print(weight, height, channel)
    fps = 10
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 用于mp4格式的生成
    videowriter = cv2.VideoWriter(output, fourcc, fps, (weight, height))  # 创建一个写入视频对象
    for img in img_list:
        path = file_path + '/' + img
        print(path)
        # print(path)
        frame = cv2.imread(path)
        videowriter.write(frame)

    videowriter.release()

# if __name__ == '__main__':
#     pass
    # nusc = NuScenes(version='v1.0-mini', dataroot=IMGS_PATH, verbose=True)
    # frame_idx = 0
    # scene_idx = 0
    # for sample in nusc.sample:
    #     if frame_idx == 0:
    #         last_scene_token = sample['scene_token']
    #     if sample['scene_token'] != last_scene_token:
    #         frame_idx = 0 # next scene frame start from 0
    #         scene_idx += 1
    #         last_scene_token = sample['scene_token'] 
    #     frame_idx += 1
    #     mmcv.mkdir_or_exist(OUTPUT_PATH + f'/scene_{scene_idx}') 
    #     images = {'lidar': None}
    #     lidar_token = sample['data']['LIDAR_TOP'] 
    #     lidar_sample_data = nusc.get('sample_data', lidar_token)
    #     lidar_calibrated_data = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    #     ego2lidar = get_matrix(lidar_calibrated_data, inverse=True)
    #     ego_calibrated_data_l = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    #     ego2global = get_matrix(ego_calibrated_data_l, inverse=False)
    #     # boxes in lidar coordinate system
    #     lidar_path, boxes_lidar, _ = nusc.get_sample_data(lidar_token) # get_sample_data函数会将box投影到输入的token对应的传感器坐标系下
    #     # visualize lidar points
    #     image_2D = visualize_lidar_points(lidar_path, ego2lidar, pc_range, image_size)
    #     # visualize 2D boxes
    #     image_2D = visualize_2D_boxes(nusc, image_2D, boxes_lidar, pc_range, image_size)
        
    #     images["lidar"] = image_2D
        
    #     # visualize 3D on camera images
    #     cameras = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    #     for camera in cameras:
    #         camera_token = sample["data"][camera]
    #         camera_sample_data = nusc.get('sample_data', camera_token)
    #         image_path, boxes_cam, camera_intrinsic_matrix = nusc.get_sample_data(camera_token) # 3D boxes in camera coordinate system
    #         camera_intrinsic = np.eye(4)
    #         camera_intrinsic[:3, :3] = camera_intrinsic_matrix
    #         image_3D = visualize_3D_boxes(nusc, image_path, boxes_cam, camera_intrinsic, pc_range, image_size)
        
    #         images[camera] = image_3D
    #     combine_all(images, OUTPUT_PATH + f'/scene_{scene_idx}/cam_{frame_idx}.png')

        # # 设置图片文件夹路径和视频输出路径
        # image_folder = OUTPUT_PATH + f'/scene_{scene_idx}'
        # video_output = 'output_video.avi'
        # image_to_video(image_folder, video_output)

        # break
        
        
class Visualizer:
    def __init__(self,):
        pass
    
    def load_images(self, imagefiles:List[str]):
        images = []
        for imagefile in imagefiles:
            images.append(cv2.imread(imagefile))
        return images
    
    def lidar_boxes2images(self, images, lidar2imgs, boxes):
        cams = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
        for i, image in enumerate(images):
            for j in range(len(boxes)):
                box = boxes[j]
                corners = box.corners[0].numpy() # 8 x 3
                corners = np.concatenate([corners, np.ones((len(corners), 1))], axis = 1) @ lidar2imgs[i].T # 8 x 4
                if np.all(corners[:, 2] > 0): 
                    corners[:, :2] /= corners[:, [2]]
                    # if box.name in NameMapping:
                    #     name = NameMapping[box.name]
                    #     color = OBJECT_PALETTE[name]
                    color = (100, 38, 40)
                    for start, end in [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7)]:
                        cv2.line(image, tuple(corners[start][:2].astype(np.int)), tuple(corners[end][:2].astype(np.int)), color, thickness=2)
                cv2.imwrite(cams[i]+'.jpg', image)
        

         
    def visualize_lidar_points(self, points, pc_range, image_size, save=False, img_name=None, rotate_matrix=None, translation=None):
        '''
            points: np.array(n, 5) -> [x, y, z, intensity, ring_idx]
            pc_range: the range of point cloud
            img_size: the output image size
            
        '''
        points = points.reshape(-1, 5)[:, :3] # [n, 3]
        points = np.concatenate([points, np.ones((len(points), 1))], axis = 1) # [n, 4]
        points = points @ rotate_matrix.T # convert to another coordinate 
        points[:, :2] /= pc_range
        # 将pts缩放成图像大小，并平移到中心
        points[:, :2] = points[:, :2] * image_size / 2 + image_size / 2
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        for ix, iy, iz in points[:, :3]:
            if 0 <= ix < image_size and 0 <= iy <image_size:
                image[int(ix), int(iy)] = 255, 255, 255 
        if save and img_name != None:
            cv2.imwrite(img_name, image)
        else:
            return image




    
        
if __name__ == '__main__':
   past_boxes_list_in_lidar = np.array([[[-7.46657896e+02,  3.06040797e+03,  1.04500807e+02,
          6.64912283e-01,  7.28518128e-01,  1.68199277e+00,
         -4.03157737e+00],
        [-7.55171996e+02,  3.06853505e+03,  1.03908251e+02,
          4.20431435e-01,  3.44264984e-01,  2.44320607e+00,
          5.39994402e-01],
        [-7.57953276e+02,  3.07326016e+03,  1.03687947e+02,
          2.94123262e-01,  2.09379911e-01,  3.71779728e+00,
         -4.01283317e+00]],

       [[-6.92404481e+02,  3.00144732e+03,  1.05619510e+02,
          1.94541430e+00,  4.69641590e+00,  1.78325903e+00,
         -8.74893328e-01],
        [-6.92071744e+02,  3.00258097e+03,  1.05401498e+02,
          3.57569766e+00,  1.58054972e+01,  2.50653768e+00,
         -8.66450688e-01],
        [-6.92091493e+02,  3.00251400e+03,  1.04108264e+02,
          5.50584650e+00,  2.00000000e+01,  3.86182737e+00,
         -8.43809204e-01]],

       [[-7.00627113e+02,  2.99914621e+03,  1.04802355e+02,
          1.90341711e+00,  4.46162128e+00,  1.75540996e+00,
         -4.01112408e+00],
        [-7.08409794e+02,  3.00336872e+03,  1.04100987e+02,
          3.44356751e+00,  1.57455521e+01,  2.45338893e+00,
         -3.96937394e+00],
        [-7.20462624e+02,  3.01730918e+03,  1.02864394e+02,
          6.18713617e+00,  2.00000000e+01,  3.34079432e+00,
         -3.99436158e+00]],

       [[-7.69927450e+02,  3.04854281e+03,  1.02573581e+02,
          1.96188891e+00,  4.77410507e+00,  1.83286607e+00,
         -4.00677150e+00],
        [-7.79167654e+02,  3.05360091e+03,  1.02073227e+02,
          3.51628804e+00,  1.64087238e+01,  2.69121504e+00,
         -4.04736673e+00],
        [-7.79915076e+02,  3.05463301e+03,  1.02124331e+02,
          2.60705495e+00,  1.08487034e+01,  4.07299471e+00,
          1.45676915e+00]],

       [[-7.69112203e+02,  3.05514353e+03,  1.03150845e+02,
          2.06235313e+00,  5.19279814e+00,  2.08421612e+00,
         -4.00684072e+00],
        [-7.74910558e+02,  3.06043353e+03,  1.03044842e+02,
          3.49378991e+00,  1.60893192e+01,  2.96334314e+00,
         -3.99739653e+00],
        [-7.75551569e+02,  3.05972304e+03,  1.01290287e+02,
          8.12905693e+00,  2.00000000e+01,  7.65981865e+00,
         -4.02036669e+00]],

       [[-7.07676316e+02,  3.00212690e+03,  1.04480925e+02,
          1.88640118e+00,  4.49899912e+00,  1.60575414e+00,
         -3.99439812e+00],
        [-7.15005660e+02,  3.00707988e+03,  1.04039476e+02,
          3.29677439e+00,  1.49947701e+01,  2.14150453e+00,
         -3.99442366e+00],
        [-7.19484840e+02,  3.01414867e+03,  1.02770543e+02,
          6.25646114e+00,  2.00000000e+01,  3.11134005e+00,
         -3.98642466e+00]],

       [[-7.36300210e+02,  2.99600378e+03,  1.01435219e+02,
          2.35359621e+00,  5.76736307e+00,  2.53392959e+00,
         -9.38593289e-01],
        [-7.40606566e+02,  3.00250876e+03,  1.00533306e+02,
          4.47379303e+00,  1.99526348e+01,  3.69597888e+00,
          1.56525986e+00],
        [-7.46984727e+02,  3.00264750e+03,  9.84502797e+01,
          8.73788643e+00,  2.00000000e+01,  5.65599871e+00,
         -3.62029864e+00]],

       [[-7.59847716e+02,  3.06616037e+03,  1.04053714e+02,
          1.97102046e+00,  4.48708439e+00,  1.64968550e+00,
         -8.39962082e-01],
        [-7.60467295e+02,  3.06638551e+03,  1.03767774e+02,
          3.69807243e+00,  1.52618923e+01,  2.29189539e+00,
         -8.55548697e-01],
        [-7.60637766e+02,  3.06645636e+03,  1.03870602e+02,
          7.21992302e+00,  2.00000000e+01,  3.45385337e+00,
         -8.34544735e-01]],

       [[-7.57026532e+02,  3.05927144e+03,  1.03970466e+02,
          1.91556573e+00,  4.42213869e+00,  1.61422122e+00,
         -8.54116039e-01],
        [-7.58252594e+02,  3.06011970e+03,  1.03545276e+02,
          3.57458305e+00,  1.51833239e+01,  2.22336507e+00,
         -8.72684380e-01],
        [-7.60872379e+02,  3.06193643e+03,  1.03305831e+02,
          7.02174568e+00,  2.00000000e+01,  3.22192073e+00,
         -8.30378847e-01]],

       [[-7.49105984e+02,  3.02412387e+03,  1.02425067e+02,
          6.37449861e-01,  6.68210864e-01,  1.76073337e+00,
         -8.37395744e-01],
        [-7.53650756e+02,  3.02766425e+03,  1.01908677e+02,
          4.15237397e-01,  3.60146135e-01,  2.63945770e+00,
         -8.36616592e-01],
        [-7.61264072e+02,  3.03111575e+03,  1.00849829e+02,
          2.72082299e-01,  1.83303431e-01,  3.89271903e+00,
         -8.89443537e-01]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [-6.98900256e+02,  3.02385477e+03,  1.05552440e+02,
          5.15253603e-01,  1.22215998e+00,  2.06793380e+00,
         -9.85479018e-01],
        [-6.97815936e+02,  3.02621873e+03,  1.05030248e+02,
          3.80459279e-01,  1.73473394e+00,  2.61600494e+00,
         -9.66174981e-01]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [-7.53912252e+02,  3.06746539e+03,  1.04256127e+02,
          6.53926909e-01,  6.39405787e-01,  1.76789606e+00,
          1.20136039e+00],
        [-7.56971232e+02,  3.07181976e+03,  1.04241027e+02,
          4.33026761e-01,  3.53290856e-01,  2.68121195e+00,
         -4.01390247e+00]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [-7.55998621e+02,  3.06950431e+03,  1.04298017e+02,
          6.43855691e-01,  6.34995937e-01,  1.69234049e+00,
          4.76823730e-01],
        [-7.59269973e+02,  3.07463800e+03,  1.04286671e+02,
          4.45262581e-01,  3.76107216e-01,  2.53851104e+00,
         -4.01512647e+00]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [-7.05562196e+02,  3.04039540e+03,  1.06937159e+02,
          6.28495693e-01,  6.67723417e-01,  1.68523550e+00,
         -4.05065826e+00]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [-6.87322804e+02,  3.00780974e+03,  1.05460368e+02,
          7.00084627e-01,  8.26134264e-01,  1.70447481e+00,
         -8.65448615e-01]],

       [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
          0.00000000e+00],
        [-7.06403288e+02,  3.00844944e+03,  1.04290709e+02,
          1.89750850e+00,  4.57285547e+00,  1.73373532e+00,
         -3.54237395e+00]]])
   
    future_trajs_rel = np.array([[0.7285851 , 0.01134164],
       [1.403461  , 0.00168405],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ],
       [0.        , 0.        ]], dtype=float32)