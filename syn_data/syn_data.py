import os
import argparse
import cv2
import random
import numpy as np
from skimage import exposure
from scipy.interpolate import CubicSpline

def guideFilter(I, p, winSize, eps):
    mean_I = cv2.blur(I, winSize)
    mean_p = cv2.blur(p, winSize)
    mean_II = cv2.blur(I * I, winSize)
    mean_Ip = cv2.blur(I * p, winSize)
    var_I = mean_II - mean_I * mean_I
    cov_Ip = mean_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.blur(a, winSize)
    mean_b = cv2.blur(b, winSize)
    q = mean_a * I + mean_b
    return q

def syn_low(img, light, img_gray, light_max=3,
            light_min=2, noise_max=0.08, noise_min=0.03):
    light = guideFilter(light, img_gray, (3, 3), 0.01)[:, :, np.newaxis]
    n = np.random.uniform(noise_min, noise_max)
    R = img / (light + 1e-7)
    L = (light + 1e-7) ** np.random.uniform(light_min, light_max)
    return np.clip(R * L + np.random.normal(0, n, img.shape), 0, 1)

def syn_haze(img, depth, beta_max=2.0, beta_min=1.0, A_max=0.9, A_min=0.6,
             color_max=0, color_min=0):
    # Resize depth map to match the dimensions of img
    t = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Normalize t to ensure values are in [0, 1]
    t = np.clip(t, 0, 1)
    
    # Generate beta
    beta = np.random.rand(1) * (beta_max - beta_min) + beta_min
    
    # Calculate transmission using GaussianBlur with valid ksize
    ksize = (5, 5)  # Ensure both dimensions are odd and positive
    t = np.exp(-np.minimum(1 - cv2.GaussianBlur(t, ksize, 0), 0.7) * beta)

    # Generate random A values and ensure they match the shape of img
    A = np.random.rand(1) * (A_max - A_min) + A_min
    A_random = np.random.rand(3) * (color_max - color_min) + color_min
    A = A + A_random
    A = np.clip(A, A_min, A_max)

    # Combine haze effect
    return np.clip(img * t + A * (1 - t), 0, 1)

# def generate_camera_trajectory(duration, num_points=20):
#     t = np.linspace(0, duration, num_points)
#     # Simulate natural hand tremor with controlled random motion
#     x = np.random.normal(0, 2, num_points)
#     y = np.random.normal(0, 2, num_points)
#     theta = np.random.normal(0, 0.01, num_points)
    
#     # Create smooth trajectory
#     t_interp = np.linspace(0, duration, 100)
#     x_spline = CubicSpline(t, x)
#     y_spline = CubicSpline(t, y)
#     theta_spline = CubicSpline(t, theta)
    
#     return x_spline(t_interp), y_spline(t_interp), theta_spline(t_interp)

# def apply_motion_blur(img, exposure_min=0.3, exposure_max=0.7):
#     """
#     Apply realistic camera shake blur with random exposure time
#     Args:
#         img: Input image (either [0,1] float or [0,255] uint8)
#         exposure_min: Minimum exposure time
#         exposure_max: Maximum exposure time
#     Returns:
#         Blurred image in same format as input
#     """
#     # Randomize exposure time
#     exposure_time = np.random.uniform(exposure_min, exposure_max)
    
#     # Convert to float32 if needed
#     is_uint8 = img.dtype == np.uint8
#     if is_uint8:
#         img = img.astype(np.float32) / 255.0

#     h, w = img.shape[:2]
#     blurred = np.zeros_like(img, dtype=np.float32)
    
#     # Generate random camera shake trajectory
#     x_traj, y_traj, theta_traj = generate_camera_trajectory(exposure_time)
#     num_steps = len(x_traj)
    
#     # Accumulate transformed images along trajectory
#     for i in range(num_steps):
#         M = np.float32([[np.cos(theta_traj[i]), -np.sin(theta_traj[i]), x_traj[i]],
#                         [np.sin(theta_traj[i]), np.cos(theta_traj[i]), y_traj[i]]])
        
#         warped = cv2.warpAffine(img, M, (w, h))
#         blurred += warped
    
#     # Normalize
#     blurred /= num_steps
    
#     # Return in same format as input
#     if is_uint8:
#         blurred = np.clip(blurred * 255.0, 0, 255).astype(np.uint8)
#     else:
#         blurred = np.clip(blurred, 0, 1)
    
#     return blurred

def apply_motion_blur(img, size=15):
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur /= size
    return cv2.filter2D(img, -1, kernel_motion_blur)

def syn_data(hq_file, light_file, depth_file, rain_file, snow_file, out_file, 
             low, haze, rain, snow, motion_blur):
    file_list = os.listdir(hq_file)
    rain_list = os.listdir(rain_file)
    snow_list = os.listdir(snow_file)
    num_rain = random.sample(range(0, len(rain_list)), len(rain_list))
    num_snow = random.sample(range(0, len(snow_list)), len(snow_list))
    for i in range(len(file_list)):
        img = cv2.imread(hq_file + file_list[i])
        w, h, _ = img.shape
        # light = cv2.cvtColor(cv2.imread(light_file + file_list[i]), cv2.COLOR_RGB2GRAY) / 255.0
        depth = cv2.imread(depth_file + file_list[i]) / 255.0
        # rain_mask = cv2.imread(rain_file + rain_list[num_rain[i]]) / 255.0
        # rain_mask = cv2.resize(rain_mask, (h, w))
        # snow_mask = cv2.imread(snow_file + snow_list[num_snow[i]]) / 255.0
        # snow_mask = cv2.resize(snow_mask, (h, w))
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
        lq = img.copy() / 255.0
        color_dis = 1

        if low:
            lq = syn_low(lq, light, img_gray)
        if rain:
            lq = lq + rain_mask
        if snow:
            lq = lq * (1 - snow_mask) + color_dis * snow_mask
        if haze:
            lq = syn_haze(lq, depth)
        if motion_blur:
            # lq = apply_motion_blur(lq, exposure_min=0.3, exposure_max=0.7)
            lq = apply_motion_blur((lq * 255).astype(np.uint8), size=15) / 255.0

        out = lq * 255.0
        cv2.imwrite(out_file + file_list[i], out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hq-file", type=str, default='./data/clear/')
    parser.add_argument("--light-file", type=str, default='./data/light_map/')
    parser.add_argument("--depth-file", type=str, default='./data/depth_map/')
    parser.add_argument("--rain-file", type=str, default='./data/rain_mask/')
    parser.add_argument("--snow-file", type=str, default='./data/snow_mask/')
    parser.add_argument("--out-file", type=str, default='./out/')
    parser.add_argument("--low", action='store_true')
    parser.add_argument("--haze", action='store_true')
    parser.add_argument("--rain", action='store_true')
    parser.add_argument("--snow", action='store_true')
    parser.add_argument("--motion-blur", action='store_true')
    
    args = parser.parse_args()

    syn_data(args.hq_file, args.light_file, args.depth_file, args.rain_file, 
             args.snow_file, args.out_file, args.low, args.haze, args.rain, args.snow, args.motion_blur)
