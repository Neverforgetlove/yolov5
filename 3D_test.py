import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import pyrealsense2 as rs
import time
import new_detect

class Camera_test():
    def __init__(self):
        self.pc = rs.pointcloud()  # 点云
        self.points = rs.points()  # 点
        self.pipeline = rs.pipeline()  # 创建管道
        self.config = rs.config()  # 创建流式传输
        # 配置传输管道
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.pipe_profile = self.pipeline.start(self.config)  # 开启传输流
        self.align_to = rs.stream.color  # 对齐流
        self.align = rs.align(self.align_to)  # 设置为其他类型,允许和深度流对齐

        self.detect_weights = new_detect.detectapi(weights='best.pt')

    def detect_obj(self):
        points = [[[300, 200], [300, 700], [900, 700], [900, 200]]]
        points = np.asarray(points)
        while True:
            self.frames = self.pipeline.wait_for_frames()
            self.aligned_frames = self.align.process(self.frames)  # 深度图和彩色图对齐
            self.color_frame = self.aligned_frames.get_color_frame()  # 获取对齐后的彩色图
            self.depth_frame = self.aligned_frames.get_depth_frame()  # 获取对齐后的深度图

            # 获取彩色帧内参
            self.color_profile = self.color_frame.get_profile()
            self.cvs_profile = rs.video_stream_profile(self.color_profile)
            self.color_intrin = self.cvs_profile.get_intrinsics()
            self.color_intrin_part = [self.color_intrin.ppx, self.color_intrin.ppy, self.color_intrin.fx,
                                      self.color_intrin.fy]

            self.ppx = self.color_intrin_part[0]
            self.ppy = self.color_intrin_part[1]
            self.fx = self.color_intrin_part[2]
            self.fy = self.color_intrin_part[3]

            # 　图像转数组
            self.img_color = np.asanyarray(self.color_frame.get_data())
            self.img_depth = np.asanyarray(self.depth_frame.get_data())

            # 获取深度标尺
            self.depth_sensor = self.pipe_profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()

            # 　深度彩色外参,对齐
            self.depth_intrin = self.depth_frame.profile.as_video_stream_profile().intrinsics
            self.color_intrin = self.color_frame.profile.as_video_stream_profile().intrinsics
            self.depth_to_color_extrin = self.depth_frame.profile.get_extrinsics_to(self.color_frame.profile)

            cv2.drawContours(self.img_color, points, 0, (255, 0, 0), 1)

            start = time.time()

            result, names = self.detect_weights.detect([self.img_color])
            # self.img_color = result[0][0]
            for cls, (x1, y1, x2, y2), conf in result[0][1]:  # 第一张图片的处理结果标签。
                #print(cls, x1, y1, x2, y2, conf)
                if (x1 < 300 or x1 > 900) or (y1 < 200 or y1 > 700):
                    pass
                else:
                    if names[cls] == "H_LS":
                        color = (255, 0, 0)
                    elif names[cls] == "M_LS":
                        color = (0, 255, 0)
                    elif names[cls] == "L_LS":
                        color = (0, 0, 255)

                    if names[cls] == "BJJ":
                        color = [255, 0, 255]

                    if names[cls] == "LM":
                        color = [0, 255, 255]

                    cv2.rectangle(self.img_color, (x1, y1), (x2, y2), color)
                    cv2.putText(self.img_color, names[cls], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color)
                    cv2.circle(self.img_color, (int((x2+x1)/2), int((y1+y2)/2)), 1, [255, 0, 255], thickness=-1)
            cv2.imshow("vedio", self.img_color)
            end = time.time()
            seconds = end - start
            fps = 1 / seconds
            print("FPS: ", fps, seconds)
            if cv2.waitKey(1) == ord('q'):
                break
if __name__ == "__main__":
    test = Camera_test()
    test.detect_obj()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    test.pipeline.stop()
