import os
import cv2
from pathlib import Path

# 경로 설정
IMAGES_DIR = 'synthesized/images'
LABELS_DIR = 'synthesized/labels'
OUTPUT_DIR = 'synthesized/checked_images'

# bbox 그릴 색상과 두께
BOX_COLOR = (0, 255, 0)  # 초록색
BOX_THICKNESS = 2

# 출력 폴더 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_labels(label_path, img_w, img_h):
    """YOLO 포맷 라벨(txt) 파일을 읽어 절대 좌표 bbox 리스트로 변환"""
    bboxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # 잘못된 포맷 무시
            class_id, cx, cy, w, h = map(float, parts)
            # 정규화 해제
            x_center = cx * img_w
            y_center = cy * img_h
            box_w = w * img_w
            box_h = h * img_h
            # 왼쪽 위(x1, y1)와 오른쪽 아래(x2, y2) 계산
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            bboxes.append((x1, y1, x2, y2))
    return bboxes

def main():
    image_paths = list(Path(IMAGES_DIR).glob('*.png'))
    if not image_paths:
        print("이미지를 찾을 수 없습니다.")
        return

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"이미지 로드 실패: {img_path}")
            continue

        label_path = Path(LABELS_DIR) / (img_path.stem + '.txt')
        if not label_path.exists():
            print(f"라벨 파일 없음: {label_path}")
            continue

        img_h, img_w = img.shape[:2]
        bboxes = load_labels(label_path, img_w, img_h)

        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)

        # 저장
        save_path = Path(OUTPUT_DIR) / img_path.name
        cv2.imwrite(str(save_path), img)

    print(f"완료! 체크된 이미지는 {OUTPUT_DIR} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main()
