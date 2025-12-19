from http.server import BaseHTTPRequestHandler
import json
import os
import base64
from api.rate_limiter import get_client_ip, check_rate_limit, send_rate_limit_error, add_rate_limit_headers

# API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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

OCR_PROMPT = """Ты — OCR-система для распознавания юридических документов на русском языке.

ЗАДАЧА: Извлеки ВЕСЬ текст с изображения документа.

ПРАВИЛА:
1. Сохраняй оригинальную структуру документа (абзацы, списки, нумерацию)
2. Распознавай все даты, суммы, номера договоров точно
3. Сохраняй наименования организаций как есть (ООО, АО, ИП и т.д.)
4. Если текст неразборчив, отметь это как [неразборчиво]
5. НЕ добавляй никаких комментариев или пояснений — только текст документа
6. Если на изображении несколько страниц или колонок, объедини текст логически

Верни ТОЛЬКО распознанный текст документа."""


def get_mime_type(filename: str) -> str:
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
    return mime_types.get(extension, 'image/png')


def ocr_with_gemini(image_base64: str, filename: str) -> str:
    """OCR через Gemini Vision"""
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    mime_type = get_mime_type(filename)
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    image_part = {"mime_type": mime_type, "data": image_base64}

    response = model.generate_content(
        [OCR_PROMPT, image_part],
        generation_config=genai.GenerationConfig(temperature=0.1, max_output_tokens=8192),
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
        raise Exception("Не удалось распознать текст")


def ocr_with_claude(image_base64: str, filename: str) -> str:
    """OCR через Claude Vision"""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=25.0)

    mime_type = get_mime_type(filename)
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_base64}},
                {"type": "text", "text": OCR_PROMPT}
            ]
        }]
    )
    return message.content[0].text


def ocr_with_openai(image_base64: str, filename: str) -> str:
    """OCR через OpenAI Vision"""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY, timeout=25.0)

    mime_type = get_mime_type(filename)
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_base64}"}},
                {"type": "text", "text": OCR_PROMPT}
            ]
        }]
    )
    return response.choices[0].message.content


def extract_text_from_image(image_base64: str, filename: str, model: str = "gemini") -> str:
    """Извлечение текста из изображения через выбранную модель"""

    if model.startswith('claude'):
        if not ANTHROPIC_API_KEY:
            raise Exception("Claude API не настроен")
        return ocr_with_claude(image_base64, filename)
    elif model.startswith('gpt'):
        if not OPENAI_API_KEY:
            raise Exception("OpenAI API не настроен")
        return ocr_with_openai(image_base64, filename)
    else:
        # Gemini по умолчанию
        if not GEMINI_API_KEY:
            raise Exception("Gemini API не настроен")
        return ocr_with_gemini(image_base64, filename)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Rate limiting: 3 запроса в минуту с одного IP
            client_ip = get_client_ip(self.headers)
            allowed, rate_info = check_rate_limit(client_ip)
            if not allowed:
                send_rate_limit_error(self, rate_info)
                return

            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            image_base64 = data.get('image', '')
            filename = data.get('filename', 'image.png')
            model = data.get('model', 'gemini')

            if not image_base64:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Изображение не предоставлено"}).encode())
                return

            # Распознаём текст через выбранную модель
            extracted_text = extract_text_from_image(image_base64, filename, model)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            add_rate_limit_headers(self, rate_info)
            self.end_headers()
            self.wfile.write(json.dumps({"text": extracted_text, "filename": filename}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Ошибка распознавания: {str(e)}"}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
