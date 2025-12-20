from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.parse
import urllib.error

# API Keys
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Gemini setup for document parsing
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

# Промпты для разных типов документов
PARSE_PROMPTS = {
    "court_decision": """Ты — система извлечения данных из судебных решений.

Проанализируй текст судебного решения и извлеки следующую информацию в формате JSON:

{
  "title": "краткое название (например: 'Дело о взыскании неустойки по договору поставки')",
  "case_number": "номер дела (например: А40-12345/2024)",
  "court_name": "название суда",
  "decision_date": "дата решения в формате YYYY-MM-DD",
  "summary": "краткое содержание дела (2-4 предложения): о чём спор, какие обстоятельства, какое решение вынес суд",
  "key_points": ["ключевой вывод 1", "ключевой вывод 2", "ключевой вывод 3"],
  "penalty_reduced": true или false (была ли снижена неустойка по ст. 333 ГК РФ),
  "reduction_percent": число или null (на сколько процентов снижена неустойка)
}

Верни ТОЛЬКО JSON без дополнительного текста.

Текст решения:
""",

    "plenum_resolution": """Ты — система извлечения данных из постановлений Пленума Верховного Суда РФ.

Проанализируй текст постановления и извлеки следующую информацию в формате JSON:

{
  "title": "название постановления",
  "document_number": "номер постановления (например: № 7)",
  "court_name": "Пленум Верховного Суда Российской Федерации",
  "decision_date": "дата принятия в формате YYYY-MM-DD",
  "summary": "о чём постановление, какие вопросы разъясняет (2-4 предложения)",
  "key_points": ["ключевое разъяснение 1", "ключевое разъяснение 2", "ключевое разъяснение 3", "ключевое разъяснение 4", "ключевое разъяснение 5"],
  "relevant_articles": ["ст. 333 ГК РФ", "ст. 395 ГК РФ"]
}

Выдели 5-10 самых важных правовых позиций из постановления.
Верни ТОЛЬКО JSON без дополнительного текста.

Текст постановления:
""",

    "practice_review": """Ты — система извлечения данных из обзоров судебной практики.

Проанализируй текст обзора и извлеки следующую информацию в формате JSON:

{
  "title": "название обзора",
  "document_number": "номер документа (если есть)",
  "court_name": "кто издал обзор (например: Верховный Суд РФ, Арбитражный суд округа)",
  "decision_date": "дата утверждения в формате YYYY-MM-DD",
  "summary": "о чём обзор, какую практику обобщает (2-4 предложения)",
  "key_points": ["правовая позиция 1", "правовая позиция 2", "правовая позиция 3", "правовая позиция 4", "правовая позиция 5"],
  "cases_mentioned": ["А40-xxx/2023", "А56-xxx/2022"]
}

Выдели 5-10 ключевых правовых позиций из обзора.
Верни ТОЛЬКО JSON без дополнительного текста.

Текст обзора:
""",

    "scientific_article": """Ты — система извлечения данных из научных юридических статей.

Проанализируй текст статьи и извлеки следующую информацию в формате JSON:

{
  "title": "название статьи",
  "authors": "авторы статьи",
  "source": "где опубликована (журнал, сборник)",
  "decision_date": "год публикации в формате YYYY-01-01",
  "summary": "основная идея статьи, какую проблему исследует (2-4 предложения)",
  "key_points": ["ключевой тезис 1", "ключевой тезис 2", "ключевой тезис 3"],
  "practical_conclusions": ["практический вывод 1", "практический вывод 2"]
}

Выдели ключевые тезисы и практические выводы для применения в судебной практике.
Верни ТОЛЬКО JSON без дополнительного текста.

Текст статьи:
"""
}

DOCUMENT_TYPE_NAMES = {
    "court_decision": "Судебное решение",
    "plenum_resolution": "Постановление Пленума ВС РФ",
    "practice_review": "Обзор судебной практики",
    "scientific_article": "Научная статья"
}


def parse_document(text: str, doc_type: str = "court_decision") -> dict:
    """Парсинг документа с помощью Gemini"""
    if not GEMINI_API_KEY:
        raise Exception("Gemini API не настроен")

    prompt = PARSE_PROMPTS.get(doc_type, PARSE_PROMPTS["court_decision"])

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    # Ограничиваем текст
    text = text[:50000]

    response = model.generate_content(
        prompt + text,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=2048,
        ),
        safety_settings=SAFETY_SETTINGS
    )

    try:
        result_text = response.text
    except (ValueError, AttributeError):
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    result_text = candidate.content.parts[0].text
                else:
                    raise Exception("Не удалось получить ответ от AI")
        else:
            raise Exception("Не удалось получить ответ от AI")

    # Извлекаем JSON из ответа
    result_text = result_text.strip()
    if result_text.startswith('```json'):
        result_text = result_text[7:]
    if result_text.startswith('```'):
        result_text = result_text[3:]
    if result_text.endswith('```'):
        result_text = result_text[:-3]
    result_text = result_text.strip()

    try:
        parsed = json.loads(result_text)
        parsed['doc_type'] = doc_type
        parsed['doc_type_name'] = DOCUMENT_TYPE_NAMES.get(doc_type, doc_type)
        return parsed
    except json.JSONDecodeError as e:
        raise Exception(f"Ошибка парсинга JSON: {str(e)[:100]}")


def get_embedding(text: str) -> list:
    """Получить embedding через OpenAI API"""
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY не настроен")

    # Ограничиваем текст (max ~8000 токенов, для русского ~15000 символов)
    text = text[:15000]

    data = json.dumps({
        "input": text,
        "model": "text-embedding-3-small"
    }).encode('utf-8')

    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=data,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['data'][0]['embedding']
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise Exception(f"Ошибка OpenAI Embedding: {error_body[:200]}")


def supabase_request(endpoint: str, method: str = "GET", data: dict = None):
    """Выполнить запрос к Supabase"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise Exception("Supabase не настроен")

    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation"
    }

    if data:
        body = json.dumps(data).encode('utf-8')
    else:
        body = None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise Exception(f"Supabase error: {error_body}")


def search_similar_decisions(query_text: str, match_count: int = 5) -> list:
    """Поиск похожих судебных решений"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return []

    try:
        # Получаем embedding для запроса
        query_embedding = get_embedding(query_text)

        # Вызываем RPC функцию поиска
        data = json.dumps({
            "query_embedding": query_embedding,
            "match_count": match_count,
            "match_threshold": 0.5
        }).encode('utf-8')

        url = f"{SUPABASE_URL}/rest/v1/rpc/search_decisions"

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Search error: {e}")
        return []


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Получить список всех решений"""
        try:
            if not SUPABASE_URL or not SUPABASE_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Supabase не настроен"}).encode())
                return

            # Получаем все решения (без embeddings для экономии трафика)
            decisions = supabase_request(
                "court_decisions?select=id,case_number,court_name,decision_date,category,summary,penalty_reduced,reduction_percent,created_at&order=created_at.desc"
            )

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"decisions": decisions}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        """Добавить новое решение или поиск"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            action = data.get('action', 'add')

            if action == 'search':
                # Поиск похожих решений
                query = data.get('query', '')
                if not query:
                    raise Exception("Пустой поисковый запрос")

                results = search_similar_decisions(query, match_count=5)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"results": results}).encode())

            elif action == 'add':
                # Добавление нового документа
                if not SUPABASE_URL or not SUPABASE_KEY:
                    raise Exception("Supabase не настроен")

                doc_type = data.get('doc_type', 'court_decision')
                title = data.get('title', '')
                case_number = data.get('case_number', '') or data.get('document_number', '')
                court_name = data.get('court_name', '')
                decision_date = data.get('decision_date')
                category = data.get('category', doc_type)
                summary = data.get('summary', '')
                full_text = data.get('full_text', '')
                key_points = data.get('key_points', [])
                penalty_reduced = data.get('penalty_reduced', False)
                reduction_percent = data.get('reduction_percent')

                if not full_text and not summary:
                    raise Exception("Нужен текст документа или краткое содержание")

                # Создаём embedding из полного текста или summary
                text_for_embedding = full_text if full_text else summary
                embedding = get_embedding(text_for_embedding)

                # Сохраняем в базу
                new_decision = {
                    "case_number": case_number,
                    "court_name": court_name,
                    "decision_date": decision_date,
                    "category": category,
                    "summary": (title + ". " if title else "") + summary,
                    "full_text": full_text,
                    "key_points": key_points,
                    "penalty_reduced": penalty_reduced,
                    "reduction_percent": reduction_percent,
                    "embedding": embedding
                }

                result = supabase_request("court_decisions", method="POST", data=new_decision)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "message": "Решение добавлено",
                    "id": result[0]['id'] if result else None
                }).encode())

            elif action == 'delete':
                # Удаление решения
                decision_id = data.get('id')
                if not decision_id:
                    raise Exception("Не указан ID решения")

                url = f"{SUPABASE_URL}/rest/v1/court_decisions?id=eq.{decision_id}"
                headers = {
                    "apikey": SUPABASE_KEY,
                    "Authorization": f"Bearer {SUPABASE_KEY}",
                }
                req = urllib.request.Request(url, headers=headers, method="DELETE")
                urllib.request.urlopen(req, timeout=10)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "message": "Удалено"}).encode())

            elif action == 'parse':
                # Парсинг документа с помощью AI
                document_text = data.get('text', '')
                doc_type = data.get('doc_type', 'court_decision')
                if not document_text:
                    raise Exception("Текст документа не предоставлен")

                parsed_data = parse_document(document_text, doc_type)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "data": parsed_data
                }).encode())

            else:
                raise Exception(f"Неизвестное действие: {action}")

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
