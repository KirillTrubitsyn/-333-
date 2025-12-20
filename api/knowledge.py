from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.parse

# API Keys
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


def get_embedding(text: str) -> list:
    """Получить embedding через OpenAI API"""
    if not OPENAI_API_KEY:
        raise Exception("OPENAI_API_KEY не настроен")

    # Ограничиваем текст (max ~8000 токенов для embedding модели)
    text = text[:30000]

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

    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result['data'][0]['embedding']


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
                # Добавление нового решения
                if not SUPABASE_URL or not SUPABASE_KEY:
                    raise Exception("Supabase не настроен")

                case_number = data.get('case_number', '')
                court_name = data.get('court_name', '')
                decision_date = data.get('decision_date')
                category = data.get('category', 'неустойка')
                summary = data.get('summary', '')
                full_text = data.get('full_text', '')
                key_points = data.get('key_points', [])
                penalty_reduced = data.get('penalty_reduced', False)
                reduction_percent = data.get('reduction_percent')

                if not full_text and not summary:
                    raise Exception("Нужен текст решения или краткое содержание")

                # Создаём embedding из полного текста или summary
                text_for_embedding = full_text if full_text else summary
                embedding = get_embedding(text_for_embedding)

                # Сохраняем в базу
                new_decision = {
                    "case_number": case_number,
                    "court_name": court_name,
                    "decision_date": decision_date,
                    "category": category,
                    "summary": summary,
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
