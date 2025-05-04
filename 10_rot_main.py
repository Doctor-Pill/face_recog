import os
import cv2
import numpy as np

# 입력 이미지 경로
input_path = "/home/piai/바탕화면/001_preproc_hands/data_pill/data/30_1_SER_nukki_cropped.png"
output_dir = "/home/piai/바탕화면/001_preproc_hands/rotated/30"
os.makedirs(output_dir, exist_ok=True)

# 파일 이름에서 확장자 제거
base_filename = os.path.splitext(os.path.basename(input_path))[0]

# 이미지 로드
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
if img is None:
    raise ValueError("이미지를 불러올 수 없습니다.")

# 캔버스 확장
h, w = img.shape[:2]
diag = int(np.ceil(np.sqrt(h**2 + w**2)))
canvas_size = max(diag, w, h)
pad_x = (canvas_size - w) // 2
pad_y = (canvas_size - h) // 2

# 중앙 배치
canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
canvas[pad_y:pad_y+h, pad_x:pad_x+w] = img
center = (canvas_size // 2, canvas_size // 2)

# 회전 및 저장
for angle in range(0, 360, 30):
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (canvas_size, canvas_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    
    base_filename = base_filename.replace("_nukki_cropped", "")
    output_filename = f"{base_filename}_{angle:03}.png"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, rotated)
    print(f"✅ 저장: {output_path}")
