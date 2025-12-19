from http.server import BaseHTTPRequestHandler
import json
import os
import base64
from api.rate_limiter import get_client_ip, check_rate_limit, send_rate_limit_error, add_rate_limit_headers

# API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Gemini setup
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def extract_text_from_image(image_base64: str, filename: str) -> str:
    """Извлечение текста из изображения через Gemini Vision"""

    # Используем Gemini 3.0 Flash Preview для OCR
    model = genai.GenerativeModel(model_name="gemini-3.0-flash-preview")

    # Определяем MIME тип
    extension = filename.lower().split('.')[-1] if '.' in filename else 'png'
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'gif': 'image/gif',
        'webp': 'image/webp',
        'heic': 'image/heic',
        'heif': 'image/heif'
    }
    mime_type = mime_types.get(extension, 'image/png')

    # Убираем data URL prefix если есть
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    # Создаем содержимое для Vision API
    image_part = {
        "mime_type": mime_type,
        "data": image_base64
    }

    prompt = """Ты — OCR-система для распознавания юридических документов на русском языке.

ЗАДАЧА: Извлеки ВЕСЬ текст с изображения документа.

ПРАВИЛА:
1. Сохраняй оригинальную структуру документа (абзацы, списки, нумерацию)
2. Распознавай все даты, суммы, номера договоров точно
3. Сохраняй наименования организаций как есть (ООО, АО, ИП и т.д.)
4. Если текст неразборчив, отметь это как [неразборчиво]
5. НЕ добавляй никаких комментариев или пояснений — только текст документа
6. Если на изображении несколько страниц или колонок, объедини текст логически

Верни ТОЛЬКО распознанный текст документа."""

    response = model.generate_content(
        [prompt, image_part],
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=8192,
        ),
        safety_settings=SAFETY_SETTINGS
    )

    try:
        return response.text
    except (ValueError, AttributeError):
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return candidate.content.parts[0].text
        raise Exception("Не удалось распознать текст на изображении")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Rate limiting: 3 запроса в минуту с одного IP
            client_ip = get_client_ip(self.headers)
            allowed, rate_info = check_rate_limit(client_ip)
            if not allowed:
                send_rate_limit_error(self, rate_info)
                return

            if not GEMINI_API_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Gemini API не настроен. Добавьте GEMINI_API_KEY в Vercel"
                }).encode())
                return

            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            image_base64 = data.get('image', '')
            filename = data.get('filename', 'image.png')

            if not image_base64:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Изображение не предоставлено"
                }).encode())
                return

            # Распознаём текст
            extracted_text = extract_text_from_image(image_base64, filename)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            add_rate_limit_headers(self, rate_info)
            self.end_headers()
            self.wfile.write(json.dumps({
                "text": extracted_text,
                "filename": filename
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": f"Ошибка распознавания: {str(e)}"
            }).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
