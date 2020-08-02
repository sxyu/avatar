import cv2
import sys, datetime
import glob
import dlib
import math
import os
from time import sleep
from shapely.geometry import Polygon

import numpy as np

# Kinect Azure intrinsics
FX = -622.359
CX = 641.666
FY = -620.594
CY = 352.072

FRAME_W, FRAME_H = 1280, 720

GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

tracked_face_color = BLUE
new_face_color = RED
tracked_landmarks_color = BLUE
new_landmarks_color = RED

DEBUG = True
STATE_NO_FACE = 0
STATE_INIT = 1
STATE_TRACKED = 2
STATE_LOSE_TRACK_MIN = 3
STATE_LOSE_TRACK_MAX = 5
MIN_FACE_AREA=500
TRACK_BOX_WIDTH = 11

LANDMARK_OPENCV=0
LANDMARK_DLIB = 1
LANDMARK_DETECTOR=LANDMARK_DLIB

#  DATASET_PATH = '/mnt_d/Programming/0VR/OpenARK/data/avatar-dataset/car_exr/mount-tripod-loop'
DATASET_PATH = '/mnt_d/Programming/0VR/OpenARK/data/avatar-dataset/car_exr/mount-tripod-eye-open-close'
#  DATASET_PATH = '/mnt_d/Programming/0VR/OpenARK/data/avatar-dataset/car_exr/mount-tripod-real-road'

# Approximated face points. This can be replaced by real-world 3D face points
# This is using orthographic projection approximation in image coordinates
model_3D_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -40.0, -30.0),        # Mouth center
                            (-35, 55, -40.0),     # Left eye center
                            (35, 55, -40.0),      # Right eye center
                            (-25, -35, -60.0),    # Left Mouth corner
                            (25, -35, -60.0)      # Right mouth corner
                        ])

# Set fixed image resize resolution in opencv convension
#  resize_size = (640, 360)
resize_size = (960, 480)

FX_SHR = FX * (resize_size[0] / FRAME_W)
CX_SHR = CX * (resize_size[0] / FRAME_W)
FY_SHR = FY * (resize_size[1] / FRAME_H)
CY_SHR = CY * (resize_size[1] / FRAME_H)

# Approximated camera intrinsic parameters
focal_length = resize_size[0]
center = (resize_size[0]/2, resize_size[1]/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
camera_dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

#https://github.com/twairball/face_tracking/blob/master/face_tracking.py

# OpenCV Face Landmark Indexes:
# 30: nose tip; 31: right nose corner; 33: middle between nose corners; 35: left nose corner
# 36: right eye outside corner; 39: right eye inside corner; 42: left eye inside corner: 45: left eye outside corner
# 48: right mouth corner; 54: left mouth corner

def bbox_to_point(bbox):
    (bX, bY, bW, bH) = bbox
    bX, bY, bW, bH = int(bX), int(bY), int(bW), int(bH)
    return bX + (bW / 2), bY + (bH / 2), 5, 5

def dlib_full_obj_to_np(obj_detection):
    return [(p.x, p.y) for p in obj_detection.parts()]

def draw_boxes(frame, boxes, color=(0,255,0)):
    for i in range(len(boxes)):
        # Prevent empty list units
        if boxes[i]:
            (bX, bY, bW, bH) = boxes[i]
            bX, bY, bW, bH = int(bX), int(bY), int(bW), int(bH)
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), color, 1)

def draw_points(frame, points,color=(0,0,255)):
    # for set_of_landmarks in points:
    for (x, y) in points:
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)


def resize_image(image, size_limit=500.0):
    max_size = max(image.shape[0], image.shape[1])
    if max_size > size_limit:
        scale = size_limit / max_size
        _img = cv2.resize(image, None, fx=scale, fy=scale)
        return _img
    return image

#a landmark at coordinate (x, y) should have a bounding box of (x - half_width, y - half_width, half_width * 2, half_width * 2)
def make_bbox_for_landmark(landmark, half_width):
    bbox_x = int(landmark[0] - half_width)
    bbox_y = int(landmark[1] - half_width)
    return (bbox_x, bbox_y, half_width * 2 + 1, half_width * 2 + 1)

def make_bbox_from_point_list(point_list):
    # test the validity of the update
    box_min_x = math.inf
    box_min_y = math.inf
    box_max_x = 0
    box_max_y = 0
    for index in range(len(point_list)):
        # calculate its area
        if point_list[index][0]<box_min_x:
            box_min_x=point_list[index][0]
        if point_list[index][0]>box_max_x:
            box_max_x=point_list[index][0]
        if point_list[index][1]<box_min_y:
            box_min_y=point_list[index][1]
        if point_list[index][1]>box_max_y:
            box_max_y=point_list[index][1]
    
    bbox = (int(box_min_x)-5, int(box_min_y)-5, int(box_max_x-box_min_x)+10, int(box_max_y-box_min_y)+10)
    return bbox

def make_feature_bbox_from_landmarks(landmarks, feature_index=0):

    bboxes=[]
    # Define nose box
    if feature_index==0 or feature_index==1:
        point_list = landmarks[0:4]
        bbox= make_bbox_from_point_list(point_list)
        bboxes.append(bbox)

    # Define right eye box
    if feature_index==0 or feature_index==2:
        point_list = landmarks[4:10]
        bbox= make_bbox_from_point_list(point_list)
        bboxes.append(bbox)

    # Define right eye box
    if feature_index==0 or feature_index==3:
        point_list = landmarks[10:16]  
        bbox= make_bbox_from_point_list(point_list)
        bboxes.append(bbox)

    # Define right eye box
    if feature_index==0 or feature_index==4:
        point_list = landmarks[16:20]
        bbox= make_bbox_from_point_list(point_list)
        bboxes.append(bbox)

    return bboxes

def avg_dist_between_points(tracked_points, detected_points):
    if len(tracked_points) != len(detected_points):
        raise Exception
    num_points = len(tracked_points)
    total = sum([calc_distance(tracked_points[i], detected_points[i]) for i in range(len(tracked_points))])
    return total / num_points

def calc_distance(point_1, point_2):
    x1, y1 = point_1 #comes from tracker, so is x, y
    x2, y2 = point_2 #comes from detection, so is x, y
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

#return the percentage overlap between 2 face detection bounding boxes, tracked and newly detected
def overlapping_percentage(bbox_a, bbox_b):
    a_x, a_y, a_w, a_h = bbox_a #x, y is upper left corner
    b_x, b_y, b_w, b_h = bbox_b
    polygon_a = Polygon([(a_x, a_y + a_h), (a_x + a_w, a_y + a_h), (a_x + a_w, a_y), (a_x, a_y)]) #lower left, lower right, upper right, upper left
    polygon_b = Polygon([(b_x, b_y + b_h), (b_x + b_w, b_y + b_h), (b_x + b_w, b_y), (b_x, b_y)]) #lower left, lower right, upper right, upper left
    intersection = polygon_a.intersection(polygon_b)
    min_area = min(polygon_a.area, polygon_b.area)
    return float(intersection.area) / min_area

# take a bounding predicted by opencv and convert it
# to the dlib (top, right, bottom, left) 
def bb_to_rect(bb):
    top=bb[1]
    left=bb[0]
    right=bb[0]+bb[2]
    bottom=bb[1]+bb[3]
    return dlib.rectangle(left, top, right, bottom) 

# take a bounding predicted by dlib and convert it
# to the format (x, y, w, h) as we would normally do
# with OpenCV
def rect_to_bb(rect):

    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return (x, y, w, h)

class FaceDetectorDNN():

    def __init__(self, modelFile="res10_300x300_ssd_iter_140000_fp16.caffemodel", configFile="deploy.prototxt"):
        self.net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
        self.conf_threshold = .8

    def detect(self, frame):
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.net.setInput(blob)
        detections = self.net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                face = (x1, y1, x2 - x1, y2 - y1)
                faces.append(face)
        return faces

class FacemarkDetectorOpenCV():

    def __init__(self):
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel("lbfmodel.yaml")

    #faces should be an np.array of face, where face is (x, y, w, h)
    def detect(self, frame, facebox):
        #fitted[0] is a boolean representing if landmarks were found or not
        #fitted[1] is an array whose first element is a 3D np.array of points
        if len(facebox) > 0:
            faces = np.asarray([facebox])
            fitted = self.facemark.fit(frame, faces)
            success = fitted[0]
            if success:
                # landmarks annotate four track-able areas: nose(4), right eye(6), left eye(6), mouth (4)
                keep = [30, 31, 33, 35, 36, 37,38, 39, 40,41, 42, 43, 44, 45,46, 47, 48, 51, 54, 57] #indices of landmarks to keep
                self.landmarks = [fitted[1][0][0][i] for i in keep]
                return self.landmarks
            else:
                print("No success detecting")
                return []
        else:
            print("No faces passed into landmark detector")
            return []

class FacemarkDetectorDlib():

    def __init__(self):
        self.predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


    def detect(self, frame, bbox):
        if bbox:
            # reduce bbox high due to dlib training on square face boxes
            delta = bbox[3] - bbox[2]
            #  if delta>0:
            #      bbox = (bbox[0], bbox[1] + delta//2, bbox[2], bbox[3]-delta//2)
            rect = bb_to_rect(bbox)
            shape = self.predictor(frame, rect)
            points = dlib_full_obj_to_np(shape)
            # landmarks annotate four track-able areas: nose(4), right eye(6), left eye(6), mouth (4)
            keep = [30, 31, 33, 35, 36, 37, 38,39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 51, 54, 57]
            # keep = [54]
            filtered_points = [points[i] for i in keep]
            return filtered_points
        else:
            return []

class Tracker():

    def __init__(self, frame, bbox):
        # Arbitrarily picked KCF tracking
        self.tracker = cv2.TrackerKCF_create() # Boosting
        self.tracker.init(frame, bbox)

    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        return ok, bbox

class Pipeline():

    def __init__(self):
        self.face_detector = FaceDetectorDNN()
        if LANDMARK_DETECTOR==LANDMARK_OPENCV:
            self.facemark_detector = FacemarkDetectorOpenCV()
        else:
            self.facemark_detector = FacemarkDetectorDlib()
        self.landmark_trackers = []

    #return faces and True/False if faces detected
    def detect_faces(self, frame):
        faces = self.face_detector.detect(frame)
        return faces

    def detect_landmarks(self, frame, facebox):
        if len(facebox)!=4:
            #if no faces are found, return
            return [], False
        landmarks = self.facemark_detector.detect(frame, facebox)
        return landmarks

def facial_orientation(bboxes, landmarks, xyz):
    image_points = np.array([
        (0, 0),     # Nose tip
        (0, 0),     # Mouth center
        (0, 0),     # Left eye center
        (0, 0),     # Right eye center
        (0, 0),     # Left Mouth corner
        (0, 0)      # Right mouth corner
    ], dtype="double")
                
    image_points[0] = [bboxes[0][0]+bboxes[0][2]/2, bboxes[0][1]+bboxes[0][3]/2]
    image_points[1] = [bboxes[3][0]+bboxes[3][2]/2, bboxes[3][1]+bboxes[3][3]/2]
    image_points[2] = [bboxes[1][0]+bboxes[1][2]/2, bboxes[1][1]+bboxes[1][3]/2]
    image_points[3] = [bboxes[2][0]+bboxes[2][2]/2, bboxes[2][1]+bboxes[2][3]/2]
    image_points[4] = landmarks[16]
    image_points[5] = landmarks[18]
    #  points_3d = np.zeros((6, 3));
    #  for i in range(points_3d.shape[0]):
    #      py = min(int(image_points[i,1]), xyz.shape[0]-1)
    #      px = min(int(image_points[i,0]), xyz.shape[1]-1)
    #      points_3d[i, :] = xyz[py, px]
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_3D_points, image_points, camera_matrix, camera_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    #  (success, rotation_vector, translation_vector) = cv2.solvePnP(points_3d, image_points, camera_matrix, camera_dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    camera_para = (rotation_vector, translation_vector)
    
    # The following are for debugging purposes of drawing orientation
    #  (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 200.0)]), rotation_vector, translation_vector, camera_matrix, camera_dist_coeffs)
    #  p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    # 0 nose
    # 4 R eye
    p1 = ( int(landmarks[0][0]), int(landmarks[0][1]))
    
    p1_3d = xyz[min(int(landmarks[0][1]), xyz.shape[0]-1),
                min(int(landmarks[0][0]), xyz.shape[1]-1)]
    mouth_3d = xyz[min(int(landmarks[17][1]), xyz.shape[0]-1),
                min(int(landmarks[17][0]), xyz.shape[1]-1)]
    leye_3d = xyz[min(int(landmarks[4][1]), xyz.shape[0]-1),
               min(int(landmarks[4][0]), xyz.shape[1]-1)]
    meye_3d = xyz[min(int((landmarks[4][1] + landmarks[13][1]) /2), xyz.shape[0]-1),
               min(int((landmarks[4][0] + landmarks[13][0]) /2), xyz.shape[1]-1)]
    #  meye_3d = (leye_3d + reye_3d) / 2
    v_up =  meye_3d - mouth_3d 
    v_right = meye_3d - leye_3d
              
    v_fwd = np.cross(v_up, v_right)
    norm = np.linalg.norm(v_fwd)
    if norm == 0.0:
        # Vector is zero
        p2 = p1
    else:
        v_fwd /= norm
        p2_3d = p1_3d - v_fwd * 0.15
        if p2_3d[2] == 0:
            # Invalid point
            p2 = p1
        else:
            p2 = ( int(p2_3d[0] * FX_SHR / p2_3d[2] + CX_SHR),
                   int(p2_3d[1] * FY_SHR / p2_3d[2] + CY_SHR))
    return camera_para, p1, p2

def boxes_overlap(box1, box2):
    # If two boxes overlap in image, return the smaller box that should be removed
    # otherwise, return 0
    (f1X, f1Y,f1W, f1H) = box1
    f1X_center = f1X + f1W/2.0
    f1Y_center = f1Y + f1H/2.0
    (f2X, f2Y,f2W, f2H) = box2
    f2X_center = f2X + f2W/2.0
    f2Y_center = f2Y + f2H/2.0

    is_overlap = False
    if f1X_center>=f2X and f1X_center<=f2X + f2W and f1Y_center>=f2Y and f1Y_center<=f2Y + f2H:
        # box1 center is within box2
        is_overlap = True

    if f2X_center>=f1X and f2X_center<=f1X + f1W and f2Y_center>=f1Y and f2Y_center<=f1Y + f1H:
        # box1 center is within box2
        is_overlap = True    

    if is_overlap:
        if f1W*f1H > f2W*f2H:
            return 2
        else:
            return 1
    else:
        return 0


def run():
    # init video stream
    #  video_capture = cv2.VideoCapture(0)

    # init detection pipeline
    pipeline = Pipeline()

    # init state variables
    landmarks = []
    tracked_faces = []
    tracked_faces_status = [] # this list register status of individual faces
    tracked_faces_orientation = []
    tracker_list=[]
    default_stop = False

    class mock_video_capture:
        def __init__(self, path):
            self.rgb_frames = sorted(glob.glob(os.path.join(path, 'rgb/*.jpg')))
            self.depth_frames = sorted(glob.glob(os.path.join(path, 'depth_exr/*.exr')))
            self.cols = None
            self.idx = 0
        def read(self):
            if self.idx >= len(self.rgb_frames):
                return False, None
            frame = cv2.imread(self.rgb_frames[self.idx])
            self.idx += 1
            return True, frame
        def read_depth(self):
            frame = cv2.imread(self.depth_frames[self.idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            return frame
        def read_xyz(self):
            depth = cv2.imread(self.depth_frames[self.idx-1], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            if self.cols is None:
                self.cols = np.zeros(depth.shape, dtype=np.float32)
                self.rows = np.zeros(depth.shape, dtype=np.float32)
                for i in range(depth.shape[1]):
                    self.cols[:, i] = i - CX
                for i in range(depth.shape[0]):
                    self.rows[i, :] = i - CY
                self.cols /= FX
                self.rows /= FY

            xyz_map = np.zeros((*depth.shape, 3), dtype=np.float32)
            xyz_map[:,:,0] = self.cols * depth
            xyz_map[:,:,1] = self.rows * depth
            xyz_map[:,:,2] = depth
            return xyz_map

    video_capture = mock_video_capture(DATASET_PATH)

    output_video = cv2.VideoWriter('out.mp4', 0x21, 15, resize_size, True)
    while not default_stop:
        return_value, frame = video_capture.read()
        if return_value==False:
            break
        frame_xyz = video_capture.read_xyz()
        frame = cv2.resize(frame, resize_size, interpolation = cv2.INTER_AREA)
        frame_xyz = cv2.resize(frame_xyz, resize_size, interpolation = cv2.INTER_NEAREST)

        if DEBUG:
            # if debugging, copy the current frame for augmenting visual results
            display_frame = frame

        # STAGE 1: Detection
        # On new frame, try to detect faces
        faces = pipeline.detect_faces(frame)

        if len(faces) > 0:

            # Remove impossible faces inside faces
            for face_index in range(len(faces)):
                (fX, fY, fW, fH) = faces[face_index]
                if fW*fH<MIN_FACE_AREA:
                    faces[face_index][2]=0
                    faces[face_index][3]=0
                    continue

                for overlap_face_index in range(face_index):
                    # check if face_index and overlap_face_index overlap
                    if faces[overlap_face_index][3]==0:
                        continue

                    which_overlap = boxes_overlap(faces[face_index], faces[overlap_face_index])
                        
                    if which_overlap==1:
                        faces[face_index]=[]
                        break
                    elif which_overlap==2:
                        faces[overlap_face_index]=[]

            for face_index in range(len(faces)):
                if faces[face_index]:
                    # Exist non-empty face region, marge tracked faces and possible new faces
                    for tracked_face_index in range(len(tracked_faces)):
                        which_overlap = boxes_overlap(faces[face_index], tracked_faces[tracked_face_index])
                        if which_overlap!=0:
                            # It overlaps, use new face region to replace the old one
                            tracked_faces[tracked_face_index] = faces[face_index]
                            # Then remove faces entry from initialization
                            faces[face_index]=[]
                            break

            #  if DEBUG:
                # augment detected faces in the frame
                #  draw_boxes(display_frame, faces, new_face_color)
                #  draw_boxes(display_frame, tracked_faces, tracked_face_color)

        # Stage 2: Initialize New Trackers. When new face detected, initiate tracker(s)
        for face_index in range(len(faces)):

            # First ignore empty face regions
            if not faces[face_index]:
                continue

            # Second, detect new facial landmarks
            landmarks= pipeline.detect_landmarks(frame, faces[face_index])
            if len(landmarks)>0:
                # valid landmarks detected, this is a face region that can be tracked                    
                # Initiate trackers
                # bboxes = [make_bbox_for_landmark(landmark, TRACK_BOX_WIDTH) for landmark in landmarks]
                bboxes = make_feature_bbox_from_landmarks(landmarks)
                feature_trackers = [Tracker(frame, bbox) for bbox in bboxes]

                tracker_list.append(feature_trackers)
                tracked_faces.append(faces[face_index])
                tracked_faces_status.append(STATE_INIT)
                
                # Calculate face orientation
                camera_para, p1, p2 = facial_orientation(bboxes, landmarks, frame_xyz)
                tracked_faces_orientation.append(camera_para)

                if DEBUG:
                    # draw detected landmarks
                    #  draw_boxes(display_frame, bboxes, new_landmarks_color)
                    draw_points(display_frame, landmarks, RED)
                    cv2.line(display_frame, p1, p2, (255,0,0), 2)


        # Stage 3: update old tracker parameters
        if  tracked_faces:
            for face_index in range(len(tracked_faces)):
                if tracked_faces_status[face_index] == STATE_INIT:
                    # This is a new face just added above, start tracking next frame
                    tracked_faces_status[face_index] = STATE_TRACKED
                else:
                    # There are old tracked face regions, update them
                    current_track_state = tracked_faces_status[face_index]

                    feature_updates = [t.update(frame) for t in tracker_list[face_index]]
                    new_landmarks= pipeline.detect_landmarks(frame, tracked_faces[face_index])

                    if DEBUG:
                        draw_points(display_frame, new_landmarks, RED)

                    total_feature_area = 0
                    bboxes = []
                    for update_index in range(len(feature_updates)):
                        if feature_updates[update_index][0] == False:
                            # Use the new landmark to replace a failed tracked landmark
                            # tracked_landmark_list[face_index][update_index] = new_landmarks[update_index]
                            
                            # add one penalty for mis-tracked state
                            tracked_faces_status[face_index] = current_track_state + 1 

                            # update the tracker
                            bbox = make_feature_bbox_from_landmarks(landmarks, update_index+1)
                            bbox = bbox[0]
                            # bbox = make_bbox_for_landmark(new_landmarks[update_index], TRACK_BOX_WIDTH)
                            tracker_list[face_index][update_index]= Tracker(frame, bbox)

                        # else:
                        #     tracked_landmark_list[face_index][update_index] = landmark_updates[update_index][1]
                        else:
                            bbox = feature_updates[update_index][1]

                        # Check if bbox is outside the face
                            if overlapping_percentage(bbox, tracked_faces[face_index])<0.99:
                                # bbox is outside the face region, add one penalty for mis-tracked state
                                tracked_faces_status[face_index] = current_track_state + 1 

                        total_feature_area = total_feature_area + bbox[2]*bbox[3]
                        bboxes.append(bbox)

                    camera_para, p1, p2 = facial_orientation(bboxes, new_landmarks, frame_xyz)
                    tracked_faces_orientation[face_index] = camera_para

                    if DEBUG:
                        #  draw_boxes(display_frame, bboxes, tracked_landmarks_color)
                        cv2.line(display_frame, p1, p2, (255,0,0), 2)


                    if tracked_faces_status[face_index] == STATE_TRACKED:
                        # Test yet another lost track condition:
                        # If tracked region varies away from detected face region, it lost track
                        face_area = tracked_faces[face_index][2]* tracked_faces[face_index][3]
                        if total_feature_area/face_area>0.2 or total_feature_area/face_area<0.01:
                            # Deem lost track
                            tracked_faces_status[face_index] = tracked_faces_status[face_index] + 1

            # Finally, we need to remove those regions that lost track for too many frames
            face_index = 0
            while face_index<len(tracked_faces):
                if tracked_faces_status[face_index] > STATE_LOSE_TRACK_MAX:
                    tracked_faces_status.pop(face_index)
                    tracked_faces.pop(face_index)
                    tracker_list.pop(face_index)
                    tracked_faces_orientation.pop(face_index)
                else:
                    face_index = face_index + 1
 

        if DEBUG:
            # if debugging, display display_frame
            cv2.imshow('Video', display_frame)
            output_video.write(display_frame)
            pressed_key=cv2.waitKey(10) & 0xFF
            if pressed_key==27 or pressed_key==ord('q'):
                default_stop = True
    output_video.release()


if __name__ == '__main__':
    run()
    #import timeit
    #print(timeit.repeat("run()", setup="from __main__ import run", repeat = 5, number=1))
