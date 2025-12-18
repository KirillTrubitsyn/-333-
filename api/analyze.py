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
    return f"""Ты — опытный российский юрист, специализирующийся на арбитражных спорах и взыскании неустойки.

ТВОЯ ГЛАВНАЯ ЗАДАЧА: Внимательно изучить предоставленные документы и сформулировать КОНКРЕТНЫЕ аргументы против применения статьи 333 ГК РФ, основанные НА ФАКТАХ ИЗ ЭТИХ ДОКУМЕНТОВ.

КРИТИЧЕСКИ ВАЖНО - АНАЛИЗ ДОКУМЕНТОВ:
1. ОБЯЗАТЕЛЬНО извлеки из иска: наименования сторон, суммы, даты, номер договора, период просрочки, процент неустойки по договору
2. Если есть отзыв ответчика - ОБЯЗАТЕЛЬНО проанализируй каждый его довод и дай контраргумент
3. ЦИТИРУЙ конкретные факты из документов в своих аргументах
4. Используй КОНКРЕТНЫЕ цифры из документов (суммы, проценты, сроки)
5. НЕ ПИШИ абстрактные общие фразы - только конкретика из дела

СТРУКТУРА КАЖДОГО АРГУМЕНТА:
- Заголовок (одной строкой)
- Пустая строка
- Текст с КОНКРЕТНЫМИ ссылками на факты из документов
- Ссылки на нормы права

ПРИМЕР ПЛОХОГО АРГУМЕНТА (НЕ ДЕЛАЙ ТАК):
"Ответчик не представил доказательств несоразмерности неустойки"

ПРИМЕР ХОРОШЕГО АРГУМЕНТА (ДЕЛАЙ ТАК):
"Ответчик ООО 'Ромашка' ссылается на тяжелое финансовое положение, однако в материалах дела отсутствуют бухгалтерские документы, подтверждающие данный довод. При этом сумма неустойки 150 000 руб. за период с 01.01.2025 по 01.06.2025 (152 дня) рассчитана по ставке 0,1% в день, что соответствует обычной практике делового оборота."

ЕСЛИ ОТВЕТЧИК ЗАЯВЛЯЕТ О НЕСОРАЗМЕРНОСТИ - проверь и укажи:
- Соотношение неустойки к сумме долга (в процентах)
- Сравнение со ставкой ЦБ РФ ({rates_info})
- Длительность просрочки исполнения

ФОРМАТ: Без markdown (**, ##), без нумерации. Разделяй аргументы пустой строкой."""


def build_user_prompt(claim_text: str, response_text: str, other_docs: str, comments: str) -> str:
    """Создать пользовательский промпт"""
    prompt_parts = []

    prompt_parts.append("=== ДОКУМЕНТ 1: ИСК О ВЗЫСКАНИИ НЕУСТОЙКИ ===")
    prompt_parts.append("Внимательно изучи этот документ. Извлеки: стороны, суммы, даты, договор, период просрочки.")
    prompt_parts.append("")
    prompt_parts.append(claim_text)

    if response_text and response_text.strip():
        prompt_parts.append("\n\n=== ДОКУМЕНТ 2: ОТЗЫВ/ХОДАТАЙСТВО ОТВЕТЧИКА ===")
        prompt_parts.append("ВАЖНО: Проанализируй КАЖДЫЙ довод ответчика и подготовь контраргумент!")
        prompt_parts.append("")
        prompt_parts.append(response_text)

    if other_docs and other_docs.strip():
        prompt_parts.append("\n\n=== ДОПОЛНИТЕЛЬНЫЕ ДОКУМЕНТЫ ===")
        prompt_parts.append(other_docs)

    if comments and comments.strip():
        prompt_parts.append("\n\n=== ПОЯСНЕНИЯ ПОЛЬЗОВАТЕЛЯ ===")
        prompt_parts.append(comments)

    prompt_parts.append("\n\n=== ЗАДАНИЕ ===")
    prompt_parts.append("На основе КОНКРЕТНЫХ ФАКТОВ из документов выше сформулируй 5-8 аргументов против снижения неустойки.")
    prompt_parts.append("ОБЯЗАТЕЛЬНО используй конкретные наименования сторон, суммы и даты из документов!")
    prompt_parts.append("Если ответчик привел доводы - дай контраргумент на КАЖДЫЙ из них.")

    return "\n".join(prompt_parts)


def get_response_text(response):
    """Безопасное получение текста из ответа Gemini"""
    try:
        text = response.text
        # Убираем возможные остатки markdown
        text = text.replace('**', '').replace('##', '').replace('###', '').replace('*', '')
        return text
    except ValueError:
        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
                text = text.replace('**', '').replace('##', '').replace('###', '').replace('*', '')
                return text
        if response.prompt_feedback:
            return f"Запрос заблокирован: {response.prompt_feedback}"
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
                model_name="gemini-2.0-flash",
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

            # Краткое резюме
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
