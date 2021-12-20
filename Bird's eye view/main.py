from elements.yolo import YOLO
from elements.deep_sort import DEEPSORT
from elements.perspective_transform import Perspective_Transform
from elements.assets import transform_matrix, detect_color
from arguments import Arguments
from yolov5.utils.plots import plot_one_box
from pykalman import KalmanFilter

import torch
import os
import cv2
import numpy as np
import sys
import math


def main(opt):
    # Load models
    detector = YOLO(opt.yolov5_model, opt.conf_thresh, opt.iou_thresh)
    deep_sort = DEEPSORT(opt.deepsort_config)
    perspective_transform = Perspective_Transform()

    # Video capture
    cap = cv2.VideoCapture(opt.source)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    

    # Save output
    if opt.save:
        output_name = opt.source.split('/')[-1]
        output_name = output_name.split('.')[0] + '_output.' + output_name.split('.')[-1]

        output_path = os.path.join(os.getcwd(), 'inference/output')
        os.makedirs(output_path, exist_ok=True)
        output_name = os.path.join(os.getcwd(), 'inference/output', output_name)

        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        out = cv2.VideoWriter(output_name,  
                                cv2.VideoWriter_fourcc(*'mp4v'), 
                                opt.outputfps, (int(w), int(h)))

    # Variables
    
    frame_num = 0
    lastDirection = ''
    direction = ''
    
    change = False
    
    ballCoords = (0,0)
    prevballCoords = (0,0)
    teamLabel = 'None'
    
    prevDisplace = 0
    displace = 0
    
    #Change it to np.array
    xBallRaw = []
    xBall = []
    yBallRaw = []
    yBall = []
    
    xyxy1raw = []
    xyxy2raw = []
    xyxy3raw = []
    xyxy4raw = []
    
    xyxy1 = []
    xyxy2 = []
    xyxy3 = []
    xyxy4 = []
    
    kf1 = KalmanFilter(initial_state_mean=0, n_dim_obs = 1)
    
    starting = False
    rawData = np.array([])
    processedData = np.array([])
    
    temp_text = ''
    temp_counter = 0
    temp_counter2 = 0

    # Black Image (Soccer Field)
    bg_ratio = int(np.ceil(w/(3*115)))
    gt_img = cv2.imread('./inference/black.jpg')
    gt_img = cv2.resize(gt_img,(115*bg_ratio, 74*bg_ratio))
    gt_h, gt_w, _ = gt_img.shape
    

    while(cap.isOpened()):
        
        ret, frame = cap.read()
        bg_img = gt_img.copy()

        if ret:
            main_frame = frame.copy()
            yoloOutput = detector.detect(frame)

            # Output: Homography Matrix and Warped image 
            if frame_num % 5 ==0: # Calculate the homography matrix every 5 frames
                M, warped_image = perspective_transform.homography_matrix(main_frame)

            if yoloOutput:
            	
                
                distance = 500.0
                histDist = 500.0
                
                # Tracking
                deep_sort.detection_to_deepsort(yoloOutput, frame)
                
                # The homography matrix is applied to the center of the lower side of the bbox.
                for i, obj in enumerate(yoloOutput):
                    xyxy = [obj['bbox'][0][0], obj['bbox'][0][1], obj['bbox'][1][0], obj['bbox'][1][1]]
                    x_center = (xyxy[0] + xyxy[2])/2 
                    y_center = xyxy[3]
                    
                    
                    if obj['label'] == 'player':
                        temp_coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        
                        '''
                        if temp_counter == 0:
                            coords1 = temp_coords
                            temp_counter += 1
                            coords = temp_coords
                        if (abs(temp_coords[0]-coords1[0]) > 30) or (abs(temp_coords[1]-coords1[1]) > 30):
                            tempX = coords1[0]
                            tempY = coords1[1]
                            if temp_coords[0] > coords1[0]: tempX += 30
                            elif temp_coords[0] < coords1[0]: tempX -= 30
                            if temp_coords[1] > coords1[1]: tempY += 30
                            elif temp_coords[1] < coords1[1]: tempY -= 30
                            coords = (tempX, tempY)
                            coords1 = coords
                        else:
                            coords = temp_coords
                        '''
                        coords = temp_coords

                        # Test
                        #print(coords[1])
                        try:
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            color = detect_color(main_frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]])
                            cv2.circle(bg_img, coords, bg_ratio + 1, color, -1)
                            if color[1] >= 250 and color[2] >=250 :
                            	cv2.putText(bg_img, 'ENG', coords, font, 0.5, (0,0,255),1,cv2.LINE_AA)
                            	cv2.circle(bg_img, coords, bg_ratio + 1, color, -1)
                            	plot_one_box(xyxy, frame, (255, 255, 255), label="ENG")
                            	if ballCoords != (0,0):
                            	    distance = math.sqrt((coords[0]-ballCoords[0])**2 + (coords[1]-ballCoords[1])**2)
                            	    teamLabel = 'ENG'
                            	    
                            	
                            	# For attacking & defensing
                            	
                            elif color[0] < 100 and color[1] >= 100 and color[2] < 100:
                                cv2.putText(bg_img, 'ENG', coords, font, 0.5, (255,255,255),1,cv2.LINE_AA)
                                cv2.circle(bg_img, coords, bg_ratio + 1, color, -1)
                                plot_one_box(xyxy, frame, (255, 255, 255), label="ENG")
                                if ballCoords != (0,0):
                                    distance = math.sqrt((coords[0]-ballCoords[0])**2 + (coords[1]-ballCoords[1])**2)
                                    teamLabel = 'ENG'
                            	
                            elif color[0] >= 80 and color[2] < 80:
                            	cv2.putText(bg_img, 'ITA', coords, font, 0.5, (255,255,255),1,cv2.LINE_AA)
                            	cv2.circle(bg_img, coords, bg_ratio + 1, color, -1)
                            	plot_one_box(xyxy, frame, (192, 0, 0), label="ITA")
                            	if ballCoords != (0,0):
                            	    distance = math.sqrt((coords[0]-ballCoords[0])**2 + (coords[1]-ballCoords[1])**2)
                            	    teamLabel = 'ITA'
                            	    
                            	
                            #Temporal fixed for yellow and grey
                            elif color == (128, 128, 128):
                            	cv2.putText(bg_img, 'ITA', coords, font, 0.5, (255,255,255),1,cv2.LINE_AA)
                            	cv2.circle(bg_img, coords, bg_ratio + 1, (192,0,0) , -1)
                            	plot_one_box(xyxy, frame, (192, 0, 0), label="ITA")
                            	if ballCoords != (0,0):
                            	    distance = math.sqrt((coords[0]-ballCoords[0])**2 + (coords[1]-ballCoords[1])**2)
                            	    teamLabel = 'ITA'
                            	    
                            	
                            elif color == (0, 192, 192):
                            	cv2.putText(bg_img, 'ITA', coords, font, 0.5, (255,255,255),1,cv2.LINE_AA)
                            	cv2.circle(bg_img, coords, bg_ratio + 1, (192,0,0) , -1)
                            	plot_one_box(xyxy, frame, (192, 0, 0), label="ITA")
                            	if ballCoords != (0,0):
                            	    distance = math.sqrt((coords[0]-ballCoords[0])**2 + (coords[1]-ballCoords[1])**2)
                            	    teamLabel = 'ITA'
                            '''
                                if temp_ball[0] > 0:
                                    direction = 'Left'
                                else:
                                    direction = 'Right'
                                if temp_ball[1] > 0:
                                    direction = direction + 'Down'
                                else:
                                    direction = direction + 'Up'
                                
                                if distance < histDist:
                                    if abs(temp_ball[0]) < 10 and abs(temp_ball[1]) < 10:
                                        histDist = distance
                                        temp_text = teamLabel + ' is attacking'
                            '''                          
                            if change == True:
                                if distance < histDist and distance < 4:
                                    histDist = distance
                                    temp_text = teamLabel + ' is attacking'
                            
                        except:
                          pass
                    elif obj['label'] == 'ball':
                    
                    # Applying kalman on data after transition matrix
                        coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                        
                        xBallRaw.append(coords[0])
                        yBallRaw.append(coords[1])
                        
                        if temp_counter > 10:
                            xBallRaw.pop(0)
                            yBallRaw.pop(0)

                        xBall = Kalman1D(xBallRaw, 0.15)
                        yBall = Kalman1D(yBallRaw, 0.15)
                        
                        iTemp = len(xBall) - 1
                        jTemp = len(yBall) - 1
                        
                        coords = (int(round(xBall[iTemp][0],0)),int(round(yBall[jTemp][0],0)))
                        
                        cv2.circle(bg_img, coords, bg_ratio + 1, (0, 192, 192), -1)
                        plot_one_box(xyxy, frame, (102, 0, 102), label="ball")
                        ballCoords = coords
                        if prevballCoords == (0,0):
                            prevballCoords = ballCoords
                                
                        else:
                            temp_ball = (prevballCoords[0]-ballCoords[0], prevballCoords[1]-ballCoords[1])
                            #print('tempball:', temp_ball)
                                
                            displace = math.sqrt(temp_ball[0]**2 + temp_ball[1]**2)
                            if displace > prevDisplace:
                                change = True
                            else:
                                change = False
                            prevDisplace = displace
                    
                    # Kalman filter applies on raw data
                        xyxy1raw.append(xyxy[0])
                        xyxy2raw.append(xyxy[1])
                        xyxy3raw.append(xyxy[2])
                        xyxy4raw.append(xyxy[3])
                        #np.ma.append(xyxy2raw, xyxy[1])
                        #np.ma.append(xyxy3raw, xyxy[2])
                        #np.ma.append(xyxy4raw, xyxy[3])
                        
                        #kfTest = KalmanFilter(initial_state_mean = 0, n_dim_obs=4)
                        
                        #xBall = Kalman1D(xBallRaw, 0.15)
                        #yBall = Kalman1D(yBallRaw, 0.15)
                        
                        #iTemp = len(xBall) - 1
                        #jTemp = len(yBall) - 1
                        
                        #coords = (int(round(xBall[iTemp][0],0)),int(round(yBall[jTemp][0],0)))
                            
                        temp_counter += 1
                        starting = True
                        
                        ballxyxy = xyxy
                        
                if starting == True:
                    temp_counter2 += 1
                
                # For the case that Ball missing from data
                if temp_counter < temp_counter2:
                    #rawData = np.append(rawData, ballxyxy)
                    #rawData[temp_counter2]
                    
                    #kfTest = KalmanFilter(initial_state_mean = 0, n_dim_obs=4)
                    #kfTest.em(rawData).smooth(rawData)
                    
                    #xyxy1raw.append(ballxyxy[0])
                    #xyxy2raw.append(ballxyxy[1])
                    #xyxy3raw.append(ballxyxy[2])
                    #xyxy4raw.append(ballxyxy[3])
                    
                    # Infer the next measurement by telling the last one is masked
                    
                    xyxy1, cov1 = kf1.filter(np.asarray(xyxy1raw))
                    xyxy1raw.append(ballxyxy[0])
                    tempMA1 = np.ma.masked_values(xyxy1raw, -1)
                    xyxy1, cov1 = kf1.filter_update(xyxy1[-1], cov1[-1], tempMA1[-1])
                    xyxy1raw[-1] = xyxy1
                    
                    xyxy2, cov2 = kf1.filter(np.asarray(xyxy2raw))
                    xyxy2raw.append(ballxyxy[1])
                    tempMA2 = np.ma.masked_values(xyxy2raw, -1)
                    xyxy2, cov2 = kf1.filter_update(xyxy2[-1], cov2[-1], tempMA2[-1])
                    xyxy2raw[-1] = xyxy2
                    
                    xyxy3, cov3 = kf1.filter(np.asarray(xyxy3raw))
                    xyxy3raw.append(ballxyxy[2])
                    tempMA3 = np.ma.masked_values(xyxy3raw, -1)
                    xyxy3, cov3 = kf1.filter_update(xyxy3[-1], cov3[-1], tempMA3[-1])
                    xyxy3raw[-1] = xyxy3
                    
                    xyxy4, cov4 = kf1.filter(np.asarray(xyxy4raw))
                    xyxy4raw.append(ballxyxy[3])
                    tempMA4 = np.ma.masked_values(xyxy4raw, -1)
                    xyxy4, cov4 = kf1.filter_update(xyxy4[-1], cov4[-1], tempMA4[-1])
                    xyxy4raw[-1] = xyxy4
                    
                    
                    tempXY = [xyxy1[-1],xyxy2[-1],xyxy3[-1],xyxy4[-1]]
                    
                    plot_one_box(tempXY, frame, (102, 0, 102), label="ball")
                    
                    x_center = (tempXY[0] + tempXY[2])/2 
                    y_center = tempXY[3]
                    
                    coords = transform_matrix(M, (x_center, y_center), (h, w), (gt_h, gt_w))
                    cv2.circle(bg_img, coords, bg_ratio + 1, (0, 192, 192), -1)
                    
                    temp_counter += 1
                    
                    if temp_counter > 10:
                        xyxy1raw.pop(0)
                        xyxy2raw.pop(0)
                        xyxy3raw.pop(0)
                        xyxy4raw.pop(0)
                
                        
                        
                cv2.putText(frame, temp_text, (40, 400), font, 2, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                deep_sort.deepsort.increment_ages()
                

            frame[frame.shape[0]-bg_img.shape[0]:, frame.shape[1]-bg_img.shape[1]:] = bg_img  
            
            if opt.view:
                cv2.imshow('frame',frame)
                if cv2.waitKey(1) & ord('q') == 0xFF:
                    break

            # Saving the output
            if opt.save:
                out.write(frame)

            frame_num += 1
            
            
        else:
            break

        sys.stdout.write(
            "\r[Input Video : %s] [%d/%d Frames Processed]"
            % (
                opt.source,
                frame_num,
                frame_count,
            )
        )

    if opt.save:
        print(f'\n\nOutput video has been saved in {output_path}!')
    
    cap.release()
    cv2.destroyAllWindows()
    
def Kalman1D(observations,damping=1):
    observation_covariance = damping
    initial_value_guess = observations[0]
    transition_matrix = 1
    transition_covariance = 0.1
    initial_value_guess
    kf = KalmanFilter(
            initial_state_mean=initial_value_guess,
            initial_state_covariance=observation_covariance,
            observation_covariance=observation_covariance,
            transition_covariance=transition_covariance,
            transition_matrices=transition_matrix
        )
    pred_state, state_cov = kf.smooth(observations)
    return pred_state

if __name__ == '__main__':

    opt = Arguments().parse()
    with torch.no_grad():
        main(opt)

