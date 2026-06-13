"""
PancCADx - SFT Dataset Construction

Converts structured clinical data (Excel) + EUS images into
ShareGPT-format multimodal conversations for LLaMA-Factory training.

Usage:
    python dataset_establish.py

The script interactively prompts for:
    - Field selection (all/simple)
    - Classification type (binary/5-class)
    - Data modality (image+clinical / clinical-only / image-only)
    - Whether images contain bounding box annotations
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple


# --- 1. Configuration ---

def get_base_path() -> Path:
    """Determine root directory based on operating system."""
    return Path("/path/to/your/dataset/")


# Clinical variable mappings (coded values -> natural language)
MAPPINGS = {
    "性别": {"男": "男", "女": "女"},
    "吸烟史（1=每日，2=偶尔，3=无）": {
        "1": "每日", "1.0": "每日", "2": "偶尔", "2.0": "偶尔",
        "3": "无", "3.0": "无"
    },
    "饮酒史（1=习惯，2=偶尔，3=无）": {
        "1": "习惯", "1.0": "习惯", "2": "偶尔", "2.0": "偶尔",
        "3": "无", "3.0": "无"
    },
    "病变位置（1=头颈，2=体尾，3=无，4=全胰）": {
        "1": "头颈", "1.0": "头颈", "2": "体尾", "2.0": "体尾",
        "3": "无", "3.0": "无", "4": "全胰", "4.0": "全胰"
    },
    # ... (additional mappings for symptoms, history, medications, imaging features)
}

CONFIG = {
    "hospitals": {
        '1': {"name": "Hospital-A", "excel": "hospital_a/clinical_data.xlsx",
              "images": "hospital_a/eus_images", "output": "hospital_a/dataset.json"},
        '2': {"name": "Hospital-B", "excel": "hospital_b/clinical_data.xlsx",
              "images": "hospital_b/eus_images", "output": "hospital_b/dataset.json"},
        '3': {"name": "Hospital-C", "excel": "hospital_c/clinical_data.xlsx",
              "images": "hospital_c/eus_images", "output": "hospital_c/dataset.json"},
        '4': {"name": "Hospital-D", "excel": "hospital_d/clinical_data.xlsx",
              "images": "hospital_d/eus_images", "output": "hospital_d/dataset.json"},
    },
    "image_extensions": ('.png', '.jpg', '.jpeg', '.gif', '.bmp'),
    "fields": {
        "all": [
            "性别", "年龄", "BMI",
            "吸烟史（1=每日，2=偶尔，3=无）",
            "饮酒史（1=习惯，2=偶尔，3=无）",
            "症状", "既往病史", "用药史",
            "CT密度", "全胰肿胀", "胰腺实质萎缩",
            "MRI-T1描述", "MRI-T2描述", "DWI弥散受限",
            "胆管扩张", "胰管扩张",
            "Se-直接胆红素", "血CA19-9", "血癌胚抗原CEA",
            "血糖", "血淀粉酶", "血脂肪酶"
        ],
        "simple": ["性别", "血CA19-9", "血癌胚抗原CEA", "血淀粉酶", "血脂肪酶"]
    },
}

BASE_PATH = get_base_path()


# --- 2. Helper Functions ---

def is_null_like(value: Any) -> bool:
    """Check if value is null or placeholder."""
    return pd.isna(value) or str(value).strip() == '/'


def map_value(field_name: str, value: Any) -> str:
    """Map coded field values to natural language descriptions."""
    if is_null_like(value):
        return "null"
    value_str = str(value).strip()
    mapping_rules = MAPPINGS.get(field_name, {})
    if mapping_rules:
        return mapping_rules.get(value_str, value_str)
    try:
        return f"{float(value_str):.2f}"
    except (ValueError, TypeError):
        return value_str


def get_diagnosis(row: pd.Series, classify_type: str) -> Tuple[str, str]:
    """Get 5-class diagnosis and final binary/5-class label."""
    diagnosis_map = {1: "CA", 2: "NET", 3: "AIP", 4: "CP", 5: "SPT",
                     "CA": "CA", "NET": "NET", "AIP": "AIP", "CP": "CP", "SPT": "SPT"}
    raw_value = row.get("五分类（1=CA，2=NET，3=AIP，4=CP，5=SPT）")
    diagnosis_5_class = diagnosis_map.get(raw_value, "null")

    if classify_type == "5":
        final_diagnosis = diagnosis_5_class
    else:  # Binary classification
        final_diagnosis = "癌变" if diagnosis_5_class == "CA" else "非癌变"

    return diagnosis_5_class, final_diagnosis


def find_patient_images(patient_name: str, diagnosis: str, image_base: Path) -> List[str]:
    """Find all EUS images for a given patient."""
    category = "CA" if diagnosis == "CA" else "OTHERS"
    search_dir = image_base / category
    if not search_dir.exists():
        return []

    image_paths = []
    for folder in search_dir.iterdir():
        if folder.is_dir() and folder.name.lower().startswith(patient_name.lower()):
            all_files = sorted([
                p for p in folder.rglob("*")
                if p.suffix.lower() in CONFIG["image_extensions"]
            ])
            image_paths.extend(p.as_posix() for p in all_files)

    return sorted(list(set(image_paths)))


def create_conversation(row: pd.Series, images_num: int, params: Dict) -> List[Dict[str, str]]:
    """Generate a multimodal conversation entry from one data row."""
    data_type = params['data_type']
    has_image = images_num > 0

    # Build clinical information string
    clinical_content_str = ""
    if data_type in ['0', '1']:
        fields_to_use = CONFIG["fields"]['all' if params['establish_type'] == 'all' else 'simple']
        content_parts = []
        for field in fields_to_use:
            field_name_simple = field.split('（')[0]
            value = map_value(field, row.get(field))
            content_parts.append(f"{field_name_simple}：{value}")
        clinical_content_str = "以下为患者临床信息：" + "；".join(content_parts)

    # Build image tags
    image_tag_str = ""
    image_prompt_str = ""
    if data_type in ['0', '2'] and has_image:
        image_tag_str = "<image>"
        if params.get('with_anchor_box') == '2':
            image_prompt_str = (
                "这张图像是患者的EUS（超声内镜）检查图像。"
                "图像中的病灶位置已用红色方框标出。"
                "如果图像中没有红色方框，则代表本次检查未发现明确病灶。"
            )

    # Get diagnosis label
    _, final_diagnosis = get_diagnosis(row, params['classify_type'])

    # Compose user and assistant messages
    task_desc_parts = []
    if data_type in ['0', '2'] and has_image:
        task_desc_parts.append("这张EUS图像")
    if data_type in ['0', '1']:
        task_desc_parts.append("患者临床信息")
    task_desc = "和".join(task_desc_parts) if task_desc_parts else "信息"

    user_content = (
        f"{image_tag_str}你是一名经验丰富的胰腺癌诊断专家，"
        f"{image_prompt_str}"
        f"根据提供的{task_desc}，综合判断是什么病种？ "
        f"{clinical_content_str}"
    ).strip()

    assistant_content = f"根据{task_desc}，综合判断为{final_diagnosis}。"

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


# --- 3. Core Processing ---

def process_and_save_dataset(df: pd.DataFrame, params: Dict, image_base: Path, output_path: Path):
    """Process DataFrame and save as ShareGPT-format JSON."""
    print(f"\n--- Processing dataset, saving to: {output_path} ---")
    all_records = []
    data_type = params['data_type']

    for idx, row in df.iterrows():
        try:
            images = []
            if data_type in ['0', '2']:
                diagnosis_5_class, _ = get_diagnosis(row, params['classify_type'])
                images = find_patient_images(row["姓名"], diagnosis_5_class, image_base)

            if data_type == '1':  # Clinical only
                messages = create_conversation(row, 0, params)
                all_records.append({"messages": messages, "images": []})

            elif data_type == '0':  # Image + Clinical
                if images:
                    for img_path in images:
                        messages = create_conversation(row, 1, params)
                        all_records.append({"messages": messages, "images": [img_path]})
                else:
                    messages = create_conversation(row, 0, params)
                    all_records.append({"messages": messages, "images": []})

            elif data_type == '2':  # Image only
                for img_path in images:
                    messages = create_conversation(row, 1, params)
                    all_records.append({"messages": messages, "images": [img_path]})

        except Exception as e:
            print(f"Error processing row {idx + 1}: {e}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(f"--- Done. Generated {len(all_records)} records. ---")


def main():
    """Main workflow with interactive parameter selection."""
    params = {
        'establish_type': input("Field selection (6=simple, all=full, default=all): ") or 'all',
        'classify_type': input("Classification (2=binary, 5=5-class, default=2): ") or '2',
        'data_type': input("Data type (0=image+clinical, 1=clinical-only, 2=image-only, default=0): ") or '0',
        'with_anchor_box': input("Bounding box annotations (1=no, 2=yes, default=2): ") or '2',
    }
    if params['establish_type'] == '6':
        params['establish_type'] = 'simple'

    hospital_choice = input("Hospital (1/2/3/4): ")
    if hospital_choice not in CONFIG["hospitals"]:
        print("Invalid choice.")
        return

    hospital_info = CONFIG["hospitals"][hospital_choice]
    excel_path = BASE_PATH / hospital_info["excel"]
    image_base = BASE_PATH / hospital_info["images"]
    output_path = BASE_PATH / hospital_info["output"]

    df = pd.read_excel(excel_path, sheet_name="Sheet1")
    process_and_save_dataset(df, params, image_base, output_path)


if __name__ == "__main__":
    main()
    print("Done.")
