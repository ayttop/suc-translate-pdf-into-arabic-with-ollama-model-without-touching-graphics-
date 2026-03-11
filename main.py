import argparse
import logging
import os
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union

import fitz  # PyMuPDF
from ollama import Client
from tqdm import tqdm

# مكتبة الصور
from PIL import Image, ImageDraw, ImageFont

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pdf_translator.log", encoding='utf-8'),
    ],
)
logger = logging.getLogger("pdf_translator")

# --- إعدادات الخط ---
FONT_FILENAME = "Amiri-Regular.ttf"

class TranslationError(Exception):
    pass

class PDFError(Exception):
    pass

@dataclass
class TranslationConfig:
    source_lang: str
    target_lang: str
    model: str = "translategemma:12b"
    batch_size: int = 10
    skip_pages: List[int] = field(default_factory=list)
    max_chunk_size: int = 1500
    
    def __post_init__(self):
        if self.skip_pages is None:
            self.skip_pages = []

class PDFTranslator:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.client = Client(host='http://127.0.0.1:11434')
        
        self.font_path = self._find_font()
        
        if not self.font_path:
            logger.error("CRITICAL: Font file not found. Please place 'Amiri-Regular.ttf' next to main.py")
        else:
            logger.info(f"Font loaded: {self.font_path}")

        self._check_ollama_availability()

    def _find_font(self):
        if os.path.exists(FONT_FILENAME):
            return os.path.abspath(FONT_FILENAME)
        windir = os.environ.get("WINDIR", "C:\\Windows")
        arial = os.path.join(windir, "Fonts", "arial.ttf")
        if os.path.exists(arial):
            logger.warning("Using Arial. Arabic support may vary.")
            return arial
        return None

    def _check_ollama_availability(self):
        try:
            models_response = self.client.list()
            available_models = []
            if models_response and 'models' in models_response:
                available_models = [m['name'].strip() for m in models_response['models'] if m.get('name')]
            
            target_model = self.config.model.strip()
            if target_model not in available_models:
                logger.info(f"Pulling model {target_model}...")
                self.client.pull(target_model)
        except Exception as e:
            raise TranslationError(f"Ollama check failed: {e}")

    def translate_pdf(self, input_pdf, output_pdf):
        input_pdf = Path(input_pdf)
        output_pdf = Path(output_pdf)
        
        doc = fitz.open(input_pdf)
        
        for page_num in tqdm(range(len(doc)), desc="Translating"):
            page = doc[page_num]
            blocks = self._extract_text_blocks(page)
            
            if not blocks: continue

            translated_blocks = []
            for block in blocks:
                if len(block['text'].strip()) < 2: continue
                try:
                    translated = self._translate_text_with_chunking(block['text'])
                    translated_blocks.append({
                        "rect": fitz.Rect(self._calculate_average_bbox(block['bboxes'])),
                        "text": translated,
                        "size": block['size']
                    })
                except Exception as e:
                    logger.error(f"Translation error: {e}")

            # مسح النص القديم
            for block in blocks:
                rect = fitz.Rect(self._calculate_average_bbox(block['bboxes']))
                page.add_redact_annot(rect, fill=(1, 1, 1))
            page.apply_redactions()

            # إدراج النص الجديد
            for tb in translated_blocks:
                self._insert_text_as_image(page, tb["rect"], tb["text"], tb["size"])

        doc.save(output_pdf, garbage=4, deflate=True)
        logger.info(f"Saved: {output_pdf}")

    def _insert_text_as_image(self, page, rect, text, font_size):
        if not text.strip() or not self.font_path:
            return

        try:
            # 1. تحديد الأبعاد
            zoom = 2 # دقة الرسم
            width = int(rect.width * zoom)
            height = int(rect.height * zoom)
            
            if width <= 10 or height <= 10: return 

            # 2. تحميل الخط والحساب التلقائي للحجم
            current_font_size = font_size
            pil_font = ImageFont.truetype(self.font_path, int(current_font_size * zoom))
            
            # حساب المساحة المطلوبة
            dummy_img = Image.new("RGB", (1, 1))
            dummy_draw = ImageDraw.Draw(dummy_img)
            
            try:
                bbox = dummy_draw.multiline_textbbox((0, 0), text, font=pil_font, align="right", spacing=2)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            except:
                text_width, text_height = len(text) * current_font_size, current_font_size * 1.2

            # تصغير الخط إذا لم يناسب الصندوق
            attempts = 0
            while (text_width > width or text_height > height) and current_font_size > 4 and attempts < 15:
                current_font_size -= 1
                pil_font = ImageFont.truetype(self.font_path, int(current_font_size * zoom))
                try:
                    bbox = dummy_draw.multiline_textbbox((0, 0), text, font=pil_font, align="right", spacing=2)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                except:
                    pass
                attempts += 1

            # 3. الرسم الفعلي
            img = Image.new("RGB", (width, height), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            margin_x = 5
            margin_y = 5
            
            draw.text((width - margin_x, margin_y), text, fill=(0, 0, 0), font=pil_font, anchor="rt", align="right")

            # 4. الإدراج
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            page.insert_image(rect, stream=img_bytes.read())

        except Exception as e:
            logger.error(f"Error rendering text image: {e}")

    def _extract_text_blocks(self, page) -> List[Dict]:
        blocks = []
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if block["type"] == 0:
                full_text, bboxes, sizes = "", [], []
                for line in block["lines"]:
                    for span in line["spans"]:
                        full_text += span["text"] + " "
                        bboxes.append(span["bbox"])
                        sizes.append(span["size"])
                if full_text.strip():
                    blocks.append({
                        "text": full_text.strip(),
                        "bboxes": bboxes,
                        "size": sum(sizes) / len(sizes) if sizes else 11
                    })
        return blocks

    def _calculate_average_bbox(self, bboxes):
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        
        # --- تعديل أبعاد الصندوق ---
        
        # 1. مسافة أمان عمودية (لمنع الالتصاق بالسطر الذي فوق أو تحت)
        vertical_padding = 2.0
        y1 = y1 - vertical_padding
        
        # 2. تمديد العرض الأفقي (لزيادة عدد الكلمات في السطر)
        # القيمة الافتراضية 0. إذا وضعنا 20، سيزيد عرض الصندوق 20 نقطة.
        # هذا يسمح للنص بالبقاء في سطر واحد أطول بدلاً من النزول لسطر جديد.
        width_extension = 28.0  # <--- يمكنك تغيير هذه القيمة حسب الحاجة
        x1 = x1 + width_extension
        
        return [x0, y0, x1, y1]

    def _translate_text_with_chunking(self, text: str) -> str:
        if not text.strip(): return text
        chunks = self._split_text(text, self.config.max_chunk_size)
        translated_parts = []
        for chunk in chunks:
            translated_parts.append(self._translate_text(chunk))
        return " ".join(translated_parts)

    def _split_text(self, text, max_len):
        if len(text) <= max_len: return [text]
        parts = text.split('. ')
        chunks = []
        current_chunk = ""
        for part in parts:
            part = part + ". "
            if len(current_chunk) + len(part) < max_len:
                current_chunk += part
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = part
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _translate_text(self, text: str) -> str:
        if not text.strip(): return text
        try:
            target = "Arabic" if "ar" in self.config.target_lang.lower() else self.config.target_lang
            system_instruction = (
                f"You are an expert literary translator. Translate the following text from {self.config.source_lang} to {target}. "
                "Follow these rules strictly:\n"
                "1. Maintain the original meaning, tone, and nuance.\n"
                "2. Use fluent and natural vocabulary in the target language.\n"
                "3. Do not translate proper names (names of people, places, brands) unless there is a well-known equivalent.\n"
                "4. Do not add any explanations or notes.\n"
                "5. Output ONLY the translated text."
            )

            response = self.client.chat(
                model=self.config.model.strip(),
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": text}
                ]
            )
            
            return response.get('message', {}).get('content', text).strip()
        except Exception as e:
            logger.error(f"API Error: {e}")
            return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pdf")
    parser.add_argument("-o", "--output")
    parser.add_argument("-s", "--source", default="English")
    parser.add_argument("-t", "--target", default="Arabic")
    parser.add_argument("-m", "--model", default="translategemma:12b")
    args = parser.parse_args()

    output = args.output or Path(args.input_pdf).with_stem(f"{Path(args.input_pdf).stem}_wider")
    config = TranslationConfig(source_lang=args.source, target_lang=args.target, model=args.model)
    
    try:
        translator = PDFTranslator(config)
        translator.translate_pdf(args.input_pdf, output)
        print(f"Successfully created: {output}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()