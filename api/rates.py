from http.server import BaseHTTPRequestHandler
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta
import xml.etree.ElementTree as ET


def fetch_cbr_rates():
    """Получить ставки ЦБ РФ с официального API"""
    rates = []

    try:
        # API ЦБ РФ для ключевой ставки
        # Берем данные за последние 2 года
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

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
                date_str = date_elem.text.split('T')[0]  # Берем только дату
                rate = float(rate_elem.text.replace(',', '.'))
                rates.append({
                    "date_from": date_str,
                    "key_rate": rate
                })

        # Сортируем по дате
        rates.sort(key=lambda x: x['date_from'])

    except Exception as e:
        # Если API не работает, возвращаем актуальные данные вручную
        rates = [
            {"date_from": "2024-07-26", "key_rate": 18.0},
            {"date_from": "2024-09-16", "key_rate": 19.0},
            {"date_from": "2024-10-28", "key_rate": 21.0},
            {"date_from": "2024-12-20", "key_rate": 21.0},
        ]

    return rates


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            rates = fetch_cbr_rates()

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'public, max-age=3600')  # Кэш на 1 час
            self.end_headers()
            self.wfile.write(json.dumps(rates).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                "error": f"Не удалось загрузить ставки ЦБ: {str(e)}"
            }).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
