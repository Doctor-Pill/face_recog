import os
import cv2
import numpy as np
from pathlib import Path
from glob import glob

# 🟡 입력 폴더 경로만 설정하세요
INPUT_ROOT = '/home/piai/바탕화면/001_preproc_hands/data_pill/data/images'  # 예: '/home/user/pills_png'
OUTPUT_ROOT = './cropped_output'          # 결과 저장 경로

def crop_transparent_area(image):
    """
    알파 채널을 기준으로 투명 영역을 제외한 객체만 남겨 크롭
    """
    if image.shape[2] < 4:
        raise ValueError("알파 채널 없음")

    alpha = image[:, :, 3]
    coords = cv2.findNonZero(alpha)

    if coords is None:
        return image  # 전부 투명한 경우 원본 반환

    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

def process_all_png_recursive(input_dir, output_dir):
    """
    하위 폴더까지 재귀적으로 탐색하여 모든 PNG에 대해 투명 영역 크롭 수행
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    png_paths = list(input_dir.rglob("*.png"))

    for img_path in png_paths:
        relative_path = img_path.relative_to(input_dir)
        save_path = output_dir / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None or img.shape[2] != 4:
            print(f"[스킵] {img_path} - 알파 채널 없음 또는 이미지 로딩 실패")
            continue

        try:
            cropped_img = crop_transparent_area(img)
            cv2.imwrite(str(save_path), cropped_img)
            print(f"[완료] {relative_path}")
        except Exception as e:
            print(f"[오류] {relative_path} 처리 중 예외 발생: {e}")

# ▶ 실행
if __name__ == "__main__":
    process_all_png_recursive(INPUT_ROOT, OUTPUT_ROOT)
    print("✅ 모든 PNG 이미지 크롭 완료")
