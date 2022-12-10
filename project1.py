import cv2
import numpy as np


# preprocess image
def preprocess(im):
    # make a copy of the image
    res = im.copy()
    # convert to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # apply color correction
    res = cv2.convertScaleAbs(res, alpha=1.5, beta=10)
    # apply gaussian blur to smooth out noise
    res = cv2.GaussianBlur(res, (5, 5), 0)
    # use Canny to find edges
    res = cv2.Canny(res, 50, 100)
    # discard parts of the image that are not relevant
    trapezoid = np.array(
        [[(0, 1850), (1350, 1150), (1750, 1150), (3360, 1850)]])
    mask = cv2.fillPoly(np.zeros_like(res), trapezoid, 255)
    res = cv2.bitwise_and(res, mask)

    return res


# calculate line coordinates from rho and theta
def get_points(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 4000*(-b))
    y1 = int(y0 + 4000*(a))
    x2 = int(x0 - 4000*(-b))
    y2 = int(y0 - 4000*(a))

    return x1, y1, x2, y2


def LaneDetect(input, output="output"):
    # create capture object
    video = cv2.VideoCapture(input)

    # get video dimensions and framerate
    video_width = int(video.get(3))
    video_height = int(video.get(4))
    video_fps = video.get(cv2.CAP_PROP_FPS)

    # create video writer to save output
    writer = cv2.VideoWriter(
        output + '.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        video_fps,
        (video_width, video_height))

    # save rhos for detecting lane changes
    prev_left_lines = []
    prev_right_lanes = []

    # lane change detection variables
    lane_change_left = False
    lane_change_right = False
    lane_change_frames_shown = 0
    prev_x1_values = []

    frame_count = 0
    success = True
    while success:
        # read frame
        success, frame = video.read()

        if not success:
            break

        # preprocess frame
        preprocessed_frame = preprocess(frame)

        # use hough transform to find lines in the frame
        left_lines = cv2.HoughLines(
            preprocessed_frame, 1, np.pi/180, 100, max_theta=(7*np.pi)/18)
        right_lines = cv2.HoughLines(
            preprocessed_frame, 1, np.pi/180, 100, min_theta=(11*np.pi)/18)

        # if no lines are found in this frame, use the previous frame's lines
        if left_lines is None and len(prev_left_lines) > 0:
            left_lines = np.array([[prev_left_lines[-1][0]]])
        if right_lines is None and len(prev_right_lanes) > 0:
            right_lines = np.array([[prev_right_lanes[-1][0]]])

        # try to find lines with rho and theta that are close to the previous rho and theta
        if len(prev_left_lines) > 0:
            close_left_lines = [line for line in left_lines if abs(
                line[0][0] - prev_left_lines[-1][0][0]) < 50 and abs(
                    line[0][1] - prev_left_lines[-1][0][1]) < 0.1]
            if len(close_left_lines) > 0:
                left_lines = close_left_lines
        if len(prev_right_lanes) > 0:
            close_right_lines = [line for line in right_lines if abs(
                line[0][0] - prev_right_lanes[-1][0][0]) < 50 and abs(
                    line[0][1] - prev_right_lanes[-1][0][1]) < 0.1]
            if len(close_right_lines) > 0:
                right_lines = close_right_lines

        # find average rho and theta of left and right lines
        left_rho = np.average([line[0][0]
                               for line in left_lines]) if left_lines is not None else 0
        left_theta = np.average([line[0][1]
                                 for line in left_lines]) if left_lines is not None else 0
        right_rho = np.average([line[0][0]
                                for line in right_lines]) if right_lines is not None else 0
        right_theta = np.average([line[0][1]
                                  for line in right_lines]) if right_lines is not None else 0

        # calculate line coordinates from rho and theta
        x1, y1, x2, y2 = get_points(left_rho, left_theta)
        x3, y3, x4, y4 = get_points(right_rho, right_theta)

        # crop the top of the lines so they don't extend too far up
        if y2 < 1150:
            x2 = int(x2 + (1150-y2) * (x1-x2)/(y1-y2))
            y2 = 1150
        if y3 < 1150:
            x3 = int(x3 + (1150-y3) * (x4-x3)/(y4-y3))
            y3 = 1150

        # crop the bottom of the lines so they don't extend below the frame
        if y1 > 2200:
            x1 = int(x1 + (2200-y1) * (x2-x1)/(y2-y1))
            y1 = 2200
        if y4 > 2200:
            x4 = int(x4 + (2200-y4) * (x3-x4)/(y3-y4))
            y4 = 2200

        # create empty image for lane markings
        lanes_image = np.zeros_like(frame)

        # draw lane markings
        if left_lines is not None:
            cv2.line(lanes_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        if right_lines is not None:
            cv2.line(lanes_image, (x3, y3), (x4, y4), (255, 0, 0), 10)

        # fill the area between the lines
        if left_lines is not None and right_lines is not None:
            points = np.array(
                [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).reshape((-1, 1, 2))
            cv2.fillPoly(lanes_image, [points], (225, 225, 225))

        # detect lane change
        # find the average rho for the left and right lane and the average x1 value
        # from the 60 previous frames
        left_rho_avg = np.average(
            [line[0][0] for line in prev_left_lines[-60:]]) if len(prev_left_lines) > 0 else 0
        right_rho_avg = np.average(
            [line[0][0] for line in prev_right_lanes[-60:]]) if len(prev_right_lanes) > 0 else 0
        x1_avg = np.average(
            prev_x1_values[-60:]) if len(prev_x1_values) > 0 else 0

        # find the difference between the current rho and the average rho
        left_rho_diff = abs(left_rho-left_rho_avg)
        right_rho_diff = abs(right_rho-right_rho_avg)

        # if the difference is large, the car is changing lanes
        if not lane_change_left and not lane_change_right and len(
            prev_left_lines) > 0 and left_rho_diff > 170 and len(
                prev_right_lanes) > 0 and right_rho_diff > 170:
            # x1 is decreasing when the car is changing lanes to the left
            lane_change_left = x1 < x1_avg
            # x1 is increasing when the car is changing lanes to the right
            lane_change_right = x1 >= x1_avg
            lane_change_frames_shown = 0
            prev_left_lines = []
            prev_right_lanes = []
            prev_x1_values = []

        # save rho and theta for next frame
        prev_left_lines.append([[left_rho, left_theta]])
        prev_right_lanes.append([[right_rho, right_theta]])
        prev_x1_values.append(x1)

        # show text for 60 frames when car is changing lanes
        if lane_change_right or lane_change_left:
            if lane_change_frames_shown < 60:
                cv2.putText(frame, "Lane Change to Left" if lane_change_left else "Lane Change to Right", (1000, 500),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8)
                lane_change_frames_shown += 1
            else:
                lane_change_left = False
                lane_change_right = False
                lane_change_frames_shown = 0

        # overlay lane markings on original frame
        frame_with_lane_markings = cv2.addWeighted(
            frame, 0.8, lanes_image, 0.4, 0)

        # write frame to output file
        writer.write(frame_with_lane_markings)
        frame_count += 1

        # show progress
        if frame_count % 100 == 0:
            print("Frame: ", frame_count)


# %%
if __name__ == '__main__':
    LaneDetect("./input.mov", "output")
# %%
