import os
import random
import json
from glob import glob
from pathlib import Path

import cv2
import numpy as np

# 하이퍼파라미터
SIGMA_POSITION = 90
MEAN_SCALE = 1.0
STD_SCALE = 0.0
MOVE_STEP = 50
MAX_MOVE_TRY = 20
SAFE_DISTANCE_FACTOR = 0.8

# 입력 폴더
HANDS_DIR = 'hands_img'
PILLS_IMAGE_DIR = 'data_pill/images'
PILLS_JSON_DIR = 'data_pill/json'

# 출력 폴더
OUTPUT_IMAGE_DIR = 'synthesized/images'
OUTPUT_LABEL_DIR = 'synthesized/labels'

# class_to_id 매핑 파일
# CLASS_MAPPING_FILE = 'class_to_id.json'
# CLASS_KEY_NAME = 'dl_name'
CLASS_MAPPING_FILE = 'code_to_id.json'
CLASS_KEY_NAME = 'drug_N'


# 폴더 생성
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

def extract_pill_code(pill_filename):
    base = Path(pill_filename).stem
    code = base.split('_')[0].replace('K-', '')
    return code

def overlay_image(background, overlay, position):
    x, y = position
    h, w = overlay.shape[:2]
    bg_h, bg_w = background.shape[:2]

    x1 = max(x - w // 2, 0)
    y1 = max(y - h // 2, 0)
    x2 = min(x1 + w, bg_w)
    y2 = min(y1 + h, bg_h)

    overlay_crop = overlay[0:(y2 - y1), 0:(x2 - x1)]

    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        for c in range(3):
            background[y1:y2, x1:x2, c] = (
                alpha * overlay_crop[:, :, c] +
                (1 - alpha) * background[y1:y2, x1:x2, c]
            )
    else:
        background[y1:y2, x1:x2] = overlay_crop

    return background

def synthesize_one(hands_list, pills_list, num_pills, save_index, class_to_id_mapping):
    hand_path = random.choice(hands_list)
    hand_img = cv2.imread(hand_path, cv2.IMREAD_UNCHANGED)
    hand_h, hand_w = hand_img.shape[:2]
    center_x, center_y = hand_w // 2, hand_h // 2

    selected_pills = random.choices(pills_list, k=num_pills)
    placed_pills = []
    labels = []

    for pill_path in selected_pills:
        pill_img = cv2.imread(pill_path, cv2.IMREAD_UNCHANGED)
        if pill_img is None:
            continue

        pill_h, pill_w = pill_img.shape[:2]

        # json 읽기 → dl_name 추출
        pill_filename = Path(pill_path).name
        pill_code = extract_pill_code(pill_filename)
        stem = Path(pill_filename).stem
        if 'nukki' in stem or 'cropped' in stem:
            parts = stem.split('_')
            for i, part in enumerate(parts):
                if part in ['nukki', 'cropped']:
                    stem = '_'.join(parts[:i])
                    break

        json_folder = "K-" + pill_code + "_json"
        json_path = Path(PILLS_JSON_DIR) / json_folder / (stem + '.json')

        with open(json_path, 'r') as f:
            data = json.load(f)

        dl_name = data['images'][0][CLASS_KEY_NAME]

        class_id = class_to_id_mapping.get(dl_name, 0)

        r = max(pill_w, pill_h) // 2

        # 초기 위치 생성
        dx = int(np.random.normal(0, SIGMA_POSITION))
        dy = int(np.random.normal(0, SIGMA_POSITION))
        pos = np.array([center_x + dx, center_y + dy])

        for _ in range(MAX_MOVE_TRY):
            collision = False
            move_vector = np.array([0.0, 0.0])

            for (px, py, pr) in placed_pills:
                dist = np.hypot(px - pos[0], py - pos[1])
                min_dist = (pr + r) * SAFE_DISTANCE_FACTOR
                if dist < min_dist:
                    collision = True
                    direction = pos - np.array([px, py])
                    if np.linalg.norm(direction) == 0:
                        direction = np.random.randn(2)
                    direction = direction / np.linalg.norm(direction)
                    move_vector += direction

            if not collision:
                break

            if np.linalg.norm(move_vector) != 0:
                move_vector = move_vector / np.linalg.norm(move_vector)
                pos = pos + move_vector * MOVE_STEP

        final_pos = (int(pos[0]), int(pos[1]))
        placed_pills.append((final_pos[0], final_pos[1], r))
        hand_img = overlay_image(hand_img, pill_img, final_pos)

        # bbox: pill 이미지 크기 그대로
        bw = pill_w
        bh = pill_h
        new_cx = final_pos[0]
        new_cy = final_pos[1]

        # YOLO 포맷 (손 기준 정규화)
        norm_cx = new_cx / hand_w
        norm_cy = new_cy / hand_h
        norm_w = bw / hand_w
        norm_h = bh / hand_h

        labels.append(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

    # 저장
    filename = f"H-{save_index:06d}"
    img_save_path = os.path.join(OUTPUT_IMAGE_DIR, filename + '.png')
    label_save_path = os.path.join(OUTPUT_LABEL_DIR, filename + '.txt')

    cv2.imwrite(img_save_path, hand_img)

    with open(label_save_path, 'w') as f:
        for line in labels:
            f.write(line + '\n')

def main():
    # 데이터 리스트 불러오기
    hands_list = glob(os.path.join(HANDS_DIR, '*.jpg')) + glob(os.path.join(HANDS_DIR, '*.png'))
    
    pills_list = []
    for pill_type_dir in Path(PILLS_IMAGE_DIR).iterdir():
        if pill_type_dir.is_dir():
            pills_list.extend(glob(str(pill_type_dir / '*.jpg')))
            pills_list.extend(glob(str(pill_type_dir / '*.png')))
    
    if not hands_list or not pills_list:
        print("손 사진 또는 알약 사진이 부족합니다.")
        return

    # class_to_id.json 로드
    with open(CLASS_MAPPING_FILE, 'r', encoding='utf-8') as f:
        class_to_id_mapping = json.load(f)

    # 사용자 입력
    total_images = int(input("생성할 합성 이미지 개수: "))
    num_pills_min = int(input("합성할 최소 알약 개수: "))
    num_pills_max = int(input("합성할 최대 알약 개수: "))

    for idx in range(total_images):
        num_pills = random.randint(num_pills_min, num_pills_max)
        synthesize_one(hands_list, pills_list, num_pills, idx, class_to_id_mapping)

    print("합성이 완료되었습니다!")

if __name__ == "__main__":
    main()
