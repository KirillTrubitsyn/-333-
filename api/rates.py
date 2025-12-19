from http.server import BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET
from api.rate_limiter import get_client_ip, check_rate_limit, send_rate_limit_error, add_rate_limit_headers


def fetch_cbr_rates():
    """Получить ставки ЦБ РФ с официального API"""
    rates = []

    try:
        # API ЦБ РФ для ключевой ставки
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        url = f"https://www.cbr.ru/DailyInfoWebServ/DailyInfo.asmx/KeyRateXML?fromDate={start_date.strftime('%Y-%m-%d')}&ToDate={end_date.strftime('%Y-%m-%d')}"

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read().decode('utf-8')

        # Парсим XML
        root = ET.fromstring(xml_data)

        for kr in root.findall('.//KR'):
            date_elem = kr.find('DT')
            rate_elem = kr.find('Rate')

            if date_elem is not None and rate_elem is not None:
                date_str = date_elem.text.split('T')[0]
                rate = float(rate_elem.text.replace(',', '.'))
                rates.append({
                    "date_from": date_str,
                    "key_rate": rate
                })

        # Сортируем по дате
        rates.sort(key=lambda x: x['date_from'])

    except Exception as e:
        # Fallback - актуальные данные на декабрь 2025
        # Ключевая ставка ЦБ РФ: 16.5% с 24.10.2025
        rates = [
            {"date_from": "2025-10-24", "key_rate": 16.5},
        ]

    return rates


def get_current_rate(rates):
    """Получить текущую (последнюю) ставку"""
    if not rates:
        return {"date_from": "2025-10-24", "key_rate": 16.5}

    # Возвращаем последнюю ставку (самую актуальную)
    return rates[-1]


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Rate limiting: 3 запроса в минуту с одного IP
            client_ip = get_client_ip(self.headers)
            allowed, rate_info = check_rate_limit(client_ip)
            if not allowed:
                send_rate_limit_error(self, rate_info)
                return

            rates = fetch_cbr_rates()
            current_rate = get_current_rate(rates)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'public, max-age=3600')  # Кэш на 1 час
            add_rate_limit_headers(self, rate_info)
            self.end_headers()

            # Возвращаем текущую ставку и историю
            self.wfile.write(json.dumps({
                "current": current_rate,
                "history": rates
            }).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": f"Не удалось загрузить ставки ЦБ: {str(e)}",
                "current": {"date_from": "2025-10-24", "key_rate": 16.5}
            }).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
