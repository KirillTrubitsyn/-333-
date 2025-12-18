from http.server import BaseHTTPRequestHandler
import json
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Конфигурация Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Настройки безопасности - разрешаем юридический контент
SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


def build_system_prompt(rates_info: str) -> str:
    """Создать системный промпт для Gemini"""
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
    """Создать пользовательский промпт"""
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


def get_response_text(response):
    """Безопасное получение текста из ответа Gemini"""
    text = ""

    # Пробуем получить текст разными способами
    try:
        text = response.text
    except (ValueError, AttributeError):
        # Пробуем через candidates
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text = candidate.content.parts[0].text
        except (AttributeError, IndexError):
            pass

    # Если текст получен, очищаем от markdown
    if text:
        text = text.replace('**', '').replace('##', '').replace('###', '').replace('*', '')
        return text

    # Проверяем причину блокировки
    try:
        if response.prompt_feedback:
            return f"Запрос заблокирован: {response.prompt_feedback}"
    except AttributeError:
        pass

    return "Не удалось получить ответ от AI"


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            if not GEMINI_API_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Gemini API не настроен. Установите GEMINI_API_KEY в переменных окружения Vercel"
                }).encode())
                return

            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            claim_text = data.get('claim_text', '')
            response_text = data.get('response_text', '')
            other_documents = data.get('other_documents', '')
            user_comments = data.get('user_comments', '')
            rates_info = data.get('rates_info', 'Ставки ЦБ недоступны')

            if not claim_text.strip():
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Текст иска обязателен для анализа"
                }).encode())
                return

            system_prompt = build_system_prompt(rates_info)
            user_prompt = build_user_prompt(
                claim_text,
                response_text,
                other_documents,
                user_comments
            )

            model = genai.GenerativeModel(
                model_name="gemini-2.5-pro",
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

            arguments_text = get_response_text(response)

            # Краткое резюме (опционально, не блокирует основной результат)
            summary = ""
            try:
                summary_prompt = f"""Кратко (2-3 предложения) опиши основные аргументы. Без markdown, без звездочек:

{arguments_text[:2000]}"""

                summary_response = model.generate_content(
                    summary_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=300,
                    ),
                    safety_settings=SAFETY_SETTINGS
                )
                summary = get_response_text(summary_response)
                # Если summary содержит ошибку, просто оставим пустым
                if "Не удалось" in summary or "заблокирован" in summary:
                    summary = ""
            except Exception:
                summary = ""  # Игнорируем ошибки summary

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "arguments_text": arguments_text,
                "summary": summary
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": f"Ошибка при обращении к Gemini AI: {str(e)}"
            }).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
