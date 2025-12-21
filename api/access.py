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
        """Получить список инвайт-кодов"""
        try:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            action = params.get('action', ['list_codes'])[0]

            if action == 'list_codes':
                codes = supabase_request(
                    "invite_codes?select=id,code,used_by,created_at&order=created_at.desc"
                )
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({"codes": codes}).encode())
            else:
                raise Exception(f"Неизвестное действие: {action}")

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_POST(self):
        """Управление инвайт-кодами"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            action = data.get('action', '')

            if action == 'verify_code':
                # Проверка инвайт-кода
                code = data.get('code', '').strip()
                name = data.get('name', '').strip()

                if not code:
                    raise Exception("Код не указан")
                if not name:
                    raise Exception("Имя не указано")

                # Ищем код в базе
                codes = supabase_request(
                    f"invite_codes?code=eq.{urllib.parse.quote(code)}&select=id,code,used_by"
                )

                if not codes:
                    raise Exception("Неверный инвайт-код")

                invite = codes[0]

                # Проверяем, не использован ли уже
                if invite.get('used_by'):
                    # Код уже использован, но разрешаем повторный вход с тем же именем
                    if invite['used_by'] != name:
                        raise Exception("Этот код уже использован другим пользователем")
                else:
                    # Помечаем код как использованный
                    url = f"{SUPABASE_URL}/rest/v1/invite_codes?id=eq.{invite['id']}"
                    headers = {
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal"
                    }
                    req = urllib.request.Request(
                        url,
                        data=json.dumps({"used_by": name}).encode('utf-8'),
                        headers=headers,
                        method="PATCH"
                    )
                    urllib.request.urlopen(req, timeout=10)

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "message": "Код подтверждён"
                }).encode())

            elif action == 'create_code':
                # Создание нового кода
                code = data.get('code', '').strip()

                if not code:
                    raise Exception("Код не указан")

                # Проверяем уникальность
                existing = supabase_request(
                    f"invite_codes?code=eq.{urllib.parse.quote(code)}&select=id"
                )
                if existing:
                    raise Exception("Такой код уже существует")

                # Создаём
                result = supabase_request(
                    "invite_codes",
                    method="POST",
                    data={"code": code}
                )

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "success": True,
                    "code": code,
                    "id": result[0]['id'] if result else None
                }).encode())

            elif action == 'delete_code':
                # Удаление кода
                code = data.get('code', '').strip()

                if not code:
                    raise Exception("Код не указан")

                url = f"{SUPABASE_URL}/rest/v1/invite_codes?code=eq.{urllib.parse.quote(code)}"
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
                self.wfile.write(json.dumps({
                    "success": True,
                    "message": "Код удалён"
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
