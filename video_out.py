import cv2
import os

# Thư mục chứa frame
img_dir = "1_frames"
# Video output
output_video = "output_1.mp4"

# FPS của video
fps = 10

# Lấy danh sách ảnh (đã sort)
images = sorted([
    img for img in os.listdir(img_dir)
    if img.endswith(".jpg") or img.endswith(".png")
])

# Đọc frame đầu để lấy size
first_frame = cv2.imread(os.path.join(img_dir, images[0]))
height, width, _ = first_frame.shape

# Khởi tạo VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Ghi từng frame
for img_name in images:
    img_path = os.path.join(img_dir, img_name)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print("✅ Video saved:", output_video)
