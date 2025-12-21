from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.parse
import urllib.error

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")


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


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Получить список отзывов"""
        try:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            action = params.get('action', ['list'])[0]

            if action == 'list':
                feedbacks = supabase_request(
                    "feedback?select=id,user_name,user_role,feedback_type,page,message,status,created_at&order=created_at.desc&limit=50"
                )
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"feedbacks": feedbacks}).encode())
            else:
                raise Exception(f"Неизвестное действие: {action}")

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        """Добавить отзыв"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            user_name = data.get('user_name', 'Аноним')
            user_role = data.get('user_role', 'unknown')
            feedback_type = data.get('feedback_type', 'other')
            page = data.get('page', '')
            message = data.get('message', '').strip()

            if not message:
                raise Exception("Сообщение не может быть пустым")

            # Сохраняем в базу
            feedback_data = {
                "user_name": user_name,
                "user_role": user_role,
                "feedback_type": feedback_type,
                "page": page,
                "message": message,
                "status": "new"
            }

            result = supabase_request("feedback", method="POST", data=feedback_data)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": True,
                "message": "Отзыв сохранён",
                "id": result[0]['id'] if result else None
            }).encode())

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
