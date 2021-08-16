import numpy as np
from numpy.core.defchararray import index
import pyrealsense2.pyrealsense2 as rs
import open3d as o3d
import cv2

class Realsense ():
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        rs.config.enable_device_from_file(self.config, "RealSense.bag")
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        self.colorizer = rs.colorizer()
        self.profile = self.pipeline.start(self.config)
        self.depth_intrinsics = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth)).get_intrinsics()
        self.color_intrinsics = rs.video_stream_profile(self.profile.get_stream(rs.stream.color)).get_intrinsics() 

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        #depth_color_frame = self.colorizer.colorize(depth_frame)
        #depth_color_image = np.asanyarray(depth_color_frame.get_data())

        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image
    
    def get_aligned_frames(self):
        align = rs.align(rs.stream.color)
        frames = self.pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if not depth_frame or not color_frame:
            return False, None, None
        return True, depth_image, color_image


# 各点を画像座標に変換
def projection_cloud(cloud, intrinsics):
    height = intrinsics.height
    width  = intrinsics.width
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    # 黒単色の画像を生成
    image = np.zeros((height, width, 3), np.uint8)
    index = 0

    # 一点ずつ画素に保存
    for point in cloud.points:
        x_ = point[0]/point[2]
        y_ = point[1]/point[2]
        v = fx*x_+cx
        u = fy*y_+cy
        int_u = np.round(u).astype(int)
        int_v = np.round(v).astype(int)

        # 画角内の点をピクセルに保存
        if 0 <= int_u < height and 0 <= int_v < width:
            image.itemset((int_u,int_v,0),(cloud.colors[index][2]*255).astype(np.uint8))
            image.itemset((int_u,int_v,1),(cloud.colors[index][1]*255).astype(np.uint8))
            image.itemset((int_u,int_v,2),(cloud.colors[index][0]*255).astype(np.uint8))
        index+=1
    
    # np.savetxt('test.txt', image[:,:,0], fmt='%d')
    cv2.imshow("projection", image)
    cv2.waitKey(0)
    return image

# ピクセルごと光線に沿って探索
def projection_cloud_inverse(cloud, intrinsics):
    height = intrinsics.height
    width  = intrinsics.width
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy

    cloud_kdtree = o3d.geometry.KDTreeFlann(cloud)

    # 黒単色の画像を生成
    image = np.zeros((height, width, 3), np.uint8)
    # 探索の設定
    search_size, z_max, z_min = 0.02, 1.8, 0.3
    gyou = 0

    # 行 きったない
    for rows in image:
        retsu = 0
        print(gyou)
        for cols in rows:
            z = z_min
            while  z <= z_max:
                p_x = (retsu-int(cx))*z/fx
                p_y = (gyou-int(cy))*z/fy
                #print("search_point = ","x",p_x,"y",p_y,"z",z)
                [k,idx, _] = cloud_kdtree.search_radius_vector_3d((p_x,p_y,z),search_size)
                if 0 < k :
                    image.itemset((gyou,retsu,0),(cloud.colors[idx[0]][2]*255).astype(np.uint8))
                    image.itemset((gyou,retsu,1),(cloud.colors[idx[0]][1]*255).astype(np.uint8))
                    image.itemset((gyou,retsu,2),(cloud.colors[idx[0]][0]*255).astype(np.uint8))
                    find_flag = True
                    break
                z += search_size*2
            retsu += 1
        
        gyou+= 1
    cv2.imshow("projection_i", image)
    cv2.waitKey(0)



if __name__ == "__main__":
    print("start")
    print(cv2)
    sensor = Realsense()
    cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)
    depth_intrinsics = sensor.depth_intrinsics
    color_intrinsics = sensor.color_intrinsics
    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.set_intrinsics(color_intrinsics.width, color_intrinsics.height, color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)

    print(sensor.depth_intrinsics)
    # print(sensor.color_intrinsics)

    while True:
        ret, depth_image, color_image = sensor.get_aligned_frames()
        if ret :
            o3d_color = o3d.geometry.Image(color_image)
            o3d_depth = o3d.geometry.Image(depth_image)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, convert_rgb_to_intensity = False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam)

            cv2.imshow("Depth Stream", color_image)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('s'):
                print("saving and projection")
                o3d.io.write_point_cloud('cloud.ply', pcd)
                projection_cloud(pcd, depth_intrinsics)
                projection_cloud_inverse(pcd, depth_intrinsics)

        else:
            print("no frames")
    
