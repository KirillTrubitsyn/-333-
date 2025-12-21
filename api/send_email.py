from http.server import BaseHTTPRequestHandler
import json
import os
import urllib.request
import urllib.error

RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
ADMIN_EMAIL = "502198t@gmail.com"


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """Отправить email через Resend"""
        try:
            if not RESEND_API_KEY:
                raise Exception("RESEND_API_KEY не настроен")

            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            user_name = data.get('user_name', 'Аноним')
            user_role = data.get('user_role', 'unknown')
            feedback_type = data.get('feedback_type', 'other')
            page = data.get('page', '')
            message = data.get('message', '')

            type_names = {
                'suggestion': 'Предложение',
                'bug': 'Ошибка/баг',
                'ux': 'Улучшение интерфейса',
                'feature': 'Новая функция',
                'other': 'Другое'
            }

            role_names = {
                'admin': 'Администратор',
                'tester': 'Тестировщик'
            }

            # Формируем HTML письма
            html_content = f"""
            <h2>Новый отзыв в Anti333.AI</h2>
            <p><b>От:</b> {user_name}</p>
            <p><b>Роль:</b> {role_names.get(user_role, user_role)}</p>
            <p><b>Тип:</b> {type_names.get(feedback_type, feedback_type)}</p>
            <p><b>Страница:</b> {page}</p>
            <hr>
            <p><b>Сообщение:</b></p>
            <p style="background: #f5f5f5; padding: 15px; border-radius: 8px;">{message.replace(chr(10), '<br>')}</p>
            """

            # Отправляем через Resend
            email_data = json.dumps({
                "from": "Anti333.AI <onboarding@resend.dev>",
                "to": ADMIN_EMAIL,
                "subject": f"[Anti333] {type_names.get(feedback_type, feedback_type)} от {user_name}",
                "html": html_content
            }).encode('utf-8')

            req = urllib.request.Request(
                "https://api.resend.com/emails",
                data=email_data,
                headers={
                    "Authorization": f"Bearer {RESEND_API_KEY}",
                    "Content-Type": "application/json"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "success": True,
                "message": "Email отправлен",
                "id": result.get('id')
            }).encode())

        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": f"Resend error: {error_body}"
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
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
