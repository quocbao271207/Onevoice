import json
import base64
import os

notebook_path = '/Users/trinhquocbao/Documents/Onevoice/Onevoice_Colab_EDA.ipynb'
out_dir = '/Users/trinhquocbao/Documents/Onevoice/eda_images'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

os.makedirs(out_dir, exist_ok=True)
img_count = 0

print("=== BẮT ĐẦU TRÍCH XUẤT OUTPUT ===")
for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code':
        outputs = cell.get('outputs', [])
        if outputs:
            print(f"\n--- Output của Cell {i} ---")
        for out in outputs:
            if out.get('output_type') == 'stream':
                print("".join(out.get('text', [])))
            elif out.get('output_type') in ['display_data', 'execute_result']:
                data = out.get('data', {})
                if 'text/plain' in data:
                    print("".join(data['text/plain']))
                if 'image/png' in data:
                    img_data = data['image/png']
                    img_count += 1
                    img_path = os.path.join(out_dir, f'plot_{img_count}.png')
                    with open(img_path, 'wb') as f_img:
                        f_img.write(base64.b64decode(img_data))
                    print(f"[Đã lưu ảnh: {img_path}]")
