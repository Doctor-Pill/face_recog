import json
from pathlib import Path

# JSON 폴더 경로
PILLS_JSON_DIR = 'data_pill/json'
OUTPUT_MAPPING_FILE = 'class_to_id.json'

def generate_class_to_id_mapping():
    class_names = set()

    # 모든 _json 폴더 순회
    for json_folder in Path(PILLS_JSON_DIR).glob('*_json'):
        for json_file in json_folder.glob('*.json'):
            with open(json_file, 'r') as f:
                data = json.load(f)
            dl_name = data['images'][0]['dl_name']
            class_names.add(dl_name)

    # 이름 순서대로 정렬 후 id 부여
    class_names = sorted(class_names)
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    return class_to_id

def save_mapping(mapping, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    print(f"Class-to-ID 매핑이 {output_file} 파일로 저장되었습니다.")

if __name__ == "__main__":
    mapping = generate_class_to_id_mapping()
    save_mapping(mapping, OUTPUT_MAPPING_FILE)
