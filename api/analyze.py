from http.server import BaseHTTPRequestHandler
import json
import os
import google.generativeai as genai

# Конфигурация Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def build_system_prompt(rates_info: str) -> str:
    """Создать системный промпт для Gemini"""
    return f"""Ты — опытный российский юрист, специализирующийся на арбитражных спорах и взыскании неустойки по договорным обязательствам.

Твоя задача — на основе предоставленных документов (иска о взыскании неустойки и отзыва ответчика) подготовить ВОЗРАЖЕНИЯ ПРОТИВ ПРИМЕНЕНИЯ СТАТЬИ 333 ГК РФ.

ВАЖНО:
1. Анализируй конкретные суммы, даты и обстоятельства из предоставленных документов
2. Используй актуальные ставки ЦБ РФ для расчетов и сравнений
3. Ссылайся на Постановление Пленума ВС РФ от 24.03.2016 № 7
4. Указывай на конкретные недостатки в аргументации ответчика
5. Формируй документ в формате процессуального документа для суда

Актуальные ставки ЦБ РФ:
{rates_info}

Структура возражений:
1. Вводная часть (позиция истца)
2. Правовое обоснование недопустимости снижения неустойки
3. Анализ соразмерности неустойки (с расчетами на основе ставки ЦБ)
4. Критика доводов ответчика
5. Судебная практика
6. Заключение и просительная часть

Отвечай ТОЛЬКО на русском языке. Формируй полный текст возражений."""


def build_user_prompt(claim_text: str, response_text: str, other_docs: str, comments: str) -> str:
    """Создать пользовательский промпт"""
    prompt_parts = []

    prompt_parts.append("=== ИСК О ВЗЫСКАНИИ НЕУСТОЙКИ ===")
    prompt_parts.append(claim_text)

    if response_text and response_text.strip():
        prompt_parts.append("\n\n=== ОТЗЫВ ОТВЕТЧИКА (ХОДАТАЙСТВО О СНИЖЕНИИ НЕУСТОЙКИ) ===")
        prompt_parts.append(response_text)

    if other_docs and other_docs.strip():
        prompt_parts.append("\n\n=== ДОПОЛНИТЕЛЬНЫЕ ДОКУМЕНТЫ ===")
        prompt_parts.append(other_docs)

    if comments and comments.strip():
        prompt_parts.append("\n\n=== ПОЯСНЕНИЯ И КОММЕНТАРИИ ПОЛЬЗОВАТЕЛЯ ===")
        prompt_parts.append(comments)

    prompt_parts.append("\n\nНа основе этих документов подготовь ВОЗРАЖЕНИЯ против применения ст. 333 ГК РФ и снижения неустойки.")

    return "\n".join(prompt_parts)


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Проверяем API ключ
            if not GEMINI_API_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Gemini API не настроен. Установите GEMINI_API_KEY в переменных окружения Vercel"
                }).encode())
                return

            # Читаем тело запроса
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

            # Формируем промпты
            system_prompt = build_system_prompt(rates_info)
            user_prompt = build_user_prompt(
                claim_text,
                response_text,
                other_documents,
                user_comments
            )

            # Вызов Gemini API - модель Gemini 2.0 Flash
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=system_prompt
            )

            response = model.generate_content(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=8192,
                )
            )

            objection_text = response.text

            # Формируем краткий анализ
            summary_prompt = f"""На основе следующего текста возражений дай краткое резюме (3-5 предложений) основных аргументов:

{objection_text[:3000]}

Ответь кратко на русском языке."""

            summary_response = model.generate_content(
                summary_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                )
            )

            analysis_summary = summary_response.text

            # Отправляем ответ
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "objection_text": objection_text,
                "analysis_summary": analysis_summary
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
