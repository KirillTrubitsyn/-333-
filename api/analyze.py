from http.server import BaseHTTPRequestHandler
import json
import os
import re

# API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

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

# Claude setup - lazy import
anthropic_client = None

def get_anthropic_client():
    global anthropic_client
    if anthropic_client is None and ANTHROPIC_API_KEY:
        import anthropic
        from httpx import Timeout
        anthropic_client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=Timeout(50.0, connect=10.0)
        )
    return anthropic_client


def compress_text(text: str) -> str:
    """Сжатие текста для экономии токенов"""
    if not text:
        return ''
    # Нормализация переносов строк
    text = text.replace('\r\n', '\n')
    # Удаление множественных пустых строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Удаление множественных пробелов
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Удаление пробелов в начале и конце строк
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text.strip()


def build_system_prompt(rates_info: str) -> str:
    return f"""Ты — опытный российский юрист. Твоя задача — создать профессиональный аналитический документ в стиле Кузнецова.

СТИЛЬ КУЗНЕЦОВА — принципы:
- Формальный деловой язык без разговорных оборотов
- Безличные конструкции: «представляется обоснованным», «необходимо отметить», «следует учитывать»
- Нумерованные разделы с иерархией (1., 2., 2.1., 2.2.)
- Точные ссылки на нормы права: «п. 1 ст. 333 ГК РФ», «п. 71 Постановления Пленума ВС РФ от 24.03.2016 № 7»
- Конкретные даты (формат: 01.01.2025), суммы, проценты из документов
- Каждый тезис подкреплён фактами из дела

СТРУКТУРА ДОКУМЕНТА:

1. ФАКТИЧЕСКИЕ ОБСТОЯТЕЛЬСТВА ДЕЛА
   Краткое изложение: стороны, договор, сумма долга, период просрочки, размер неустойки.

2. ПРАВОВОЕ ОБОСНОВАНИЕ НЕДОПУСТИМОСТИ СНИЖЕНИЯ НЕУСТОЙКИ
   2.1. [Первый аргумент — заголовок]
   Текст аргумента с конкретными ссылками на факты дела и нормы права.

   2.2. [Второй аргумент — заголовок]
   ...и так далее (5-7 аргументов)

3. АНАЛИЗ ДОВОДОВ ОТВЕТЧИКА (если есть отзыв)
   3.1. Довод ответчика о [суть довода]
   Контраргумент с обоснованием.

4. ЗАКЛЮЧЕНИЕ
   Краткий вывод: оснований для применения ст. 333 ГК РФ не имеется.

ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ДАННЫЕ ИЗ ДОКУМЕНТОВ:
- Наименования сторон (ООО, АО, ИП — как в документах)
- Номер и дату договора
- Конкретные суммы (основной долг, неустойка)
- Период просрочки с датами
- Процентную ставку неустойки по договору
- Доводы ответчика (если есть отзыв) — каждый довод разбери отдельно

АКТУАЛЬНЫЕ СТАВКИ ЦБ РФ:
{rates_info}

ФОРМАТ ВЫВОДА:
- Чистый текст без markdown (без **, ##, *)
- Нумерация разделов арабскими цифрами с точкой
- Абзацы разделены пустой строкой
- Профессиональный юридический язык"""


def build_user_prompt(claim_text: str, response_text: str, other_docs: str, comments: str) -> str:
    prompt_parts = []

    prompt_parts.append("ИСХОДНЫЕ МАТЕРИАЛЫ ДЛЯ АНАЛИЗА")
    prompt_parts.append("")
    prompt_parts.append("Приложение 1. Исковое заявление о взыскании неустойки")
    prompt_parts.append("-" * 50)
    prompt_parts.append(claim_text)

    if response_text and response_text.strip():
        prompt_parts.append("")
        prompt_parts.append("Приложение 2. Отзыв ответчика / Ходатайство о применении ст. 333 ГК РФ")
        prompt_parts.append("-" * 50)
        prompt_parts.append(response_text)

    if other_docs and other_docs.strip():
        prompt_parts.append("")
        prompt_parts.append("Приложение 3. Дополнительные материалы")
        prompt_parts.append("-" * 50)
        prompt_parts.append(other_docs)

    if comments and comments.strip():
        prompt_parts.append("")
        prompt_parts.append("Указания от пользователя:")
        prompt_parts.append(comments)

    prompt_parts.append("")
    prompt_parts.append("=" * 50)
    prompt_parts.append("ЗАДАНИЕ: Составь аналитический документ в стиле Кузнецова.")
    prompt_parts.append("Извлеки из материалов все факты и создай структурированное правовое обоснование.")

    return "\n".join(prompt_parts)


def clean_markdown(text: str) -> str:
    """Убираем markdown из текста"""
    return text.replace('**', '').replace('##', '').replace('###', '').replace('*', '')


def call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Вызов Gemini API"""
    model = genai.GenerativeModel(
        model_name="gemini-3-pro-preview",
        system_instruction=system_prompt
    )

    response = model.generate_content(
        user_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.4,
            max_output_tokens=8192,
        ),
        safety_settings=SAFETY_SETTINGS
    )

    # Получаем текст
    try:
        text = response.text
    except (ValueError, AttributeError):
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text = candidate.content.parts[0].text
        except (AttributeError, IndexError):
            raise Exception("Не удалось получить ответ от Gemini")

    return clean_markdown(text)


def call_claude(system_prompt: str, user_prompt: str, model_id: str) -> str:
    """Вызов Claude API"""
    client = get_anthropic_client()
    if not client:
        raise Exception("Claude API недоступен. Проверьте ANTHROPIC_API_KEY")

    try:
        message = client.messages.create(
            model=model_id,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        text = message.content[0].text
        return clean_markdown(text)
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            raise Exception("Claude API не ответил вовремя. Попробуйте Gemini или повторите позже.")
        elif "overloaded" in error_msg:
            raise Exception("Claude API перегружен. Попробуйте позже или используйте Gemini.")
        elif "rate" in error_msg:
            raise Exception("Превышен лимит запросов Claude. Подождите минуту.")
        elif "invalid" in error_msg and "key" in error_msg:
            raise Exception("Неверный API ключ Claude.")
        elif "connection" in error_msg:
            raise Exception("Не удалось подключиться к Claude API. Проверьте интернет.")
        else:
            raise Exception(f"Ошибка Claude: {str(e)[:200]}")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            claim_text = compress_text(data.get('claim_text', ''))
            response_text = compress_text(data.get('response_text', ''))
            other_documents = compress_text(data.get('other_documents', ''))
            user_comments = data.get('user_comments', '').strip()
            rates_info = data.get('rates_info', 'Ставки ЦБ недоступны')
            model = data.get('model', 'gemini-3-pro-preview')

            if not claim_text:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Текст иска обязателен для анализа"
                }).encode())
                return

            # Проверяем доступность API
            is_claude = model.startswith('claude')
            if is_claude and not ANTHROPIC_API_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Claude API не настроен. Добавьте ANTHROPIC_API_KEY в Vercel"
                }).encode())
                return

            if not is_claude and not GEMINI_API_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Gemini API не настроен. Добавьте GEMINI_API_KEY в Vercel"
                }).encode())
                return

            system_prompt = build_system_prompt(rates_info)
            user_prompt = build_user_prompt(
                claim_text,
                response_text,
                other_documents,
                user_comments
            )

            # Вызываем нужную модель
            if is_claude:
                arguments_text = call_claude(system_prompt, user_prompt, model)
            else:
                arguments_text = call_gemini(system_prompt, user_prompt)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "arguments_text": arguments_text,
                "model_used": model
            }).encode())

        except Exception as e:
            error_msg = str(e)
            # Убираем спецсимволы которые могут сломать JSON
            error_msg = error_msg.replace('\n', ' ').replace('\r', ' ')[:500]
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"Ошибка AI: {error_msg}"
                }, ensure_ascii=False).encode('utf-8'))
            except Exception:
                # Fallback если что-то совсем пошло не так
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'{"error": "Internal server error"}')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
