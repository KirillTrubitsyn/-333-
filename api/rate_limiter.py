"""
Rate Limiter для API
Ограничивает количество запросов с одного IP адреса.

Примечание: In-memory хранилище работает в рамках одного инстанса Vercel.
Для полной надёжности на высоких нагрузках рекомендуется использовать
Upstash Redis (https://upstash.com/) или Vercel KV.
"""

import time
from collections import defaultdict
from threading import Lock

# Настройки rate limiting
RATE_LIMIT = 3  # Максимум запросов
RATE_WINDOW = 60  # Временное окно в секундах (1 минута)

# Хранилище запросов: {ip: [timestamp1, timestamp2, ...]}
request_history = defaultdict(list)
lock = Lock()


def get_client_ip(headers) -> str:
    """Получить IP клиента из заголовков запроса"""
    # Vercel передаёт реальный IP в x-forwarded-for
    forwarded_for = headers.get('X-Forwarded-For', '')
    if forwarded_for:
        # Берём первый IP из списка (реальный клиент)
        return forwarded_for.split(',')[0].strip()

    # Fallback на x-real-ip
    real_ip = headers.get('X-Real-IP', '')
    if real_ip:
        return real_ip.strip()

    # Последний fallback
    return headers.get('Host', 'unknown')


def check_rate_limit(client_ip: str) -> tuple[bool, dict]:
    """
    Проверить, не превышен ли лимит запросов.

    Returns:
        (allowed, info) - разрешён ли запрос и информация о лимитах
    """
    current_time = time.time()
    window_start = current_time - RATE_WINDOW

    with lock:
        # Удаляем старые записи за пределами окна
        request_history[client_ip] = [
            ts for ts in request_history[client_ip]
            if ts > window_start
        ]

        requests_count = len(request_history[client_ip])

        if requests_count >= RATE_LIMIT:
            # Лимит превышен
            oldest_request = min(request_history[client_ip])
            retry_after = int(oldest_request + RATE_WINDOW - current_time) + 1

            return False, {
                "allowed": False,
                "limit": RATE_LIMIT,
                "remaining": 0,
                "retry_after": max(1, retry_after),
                "window": RATE_WINDOW
            }

        # Добавляем текущий запрос
        request_history[client_ip].append(current_time)

        return True, {
            "allowed": True,
            "limit": RATE_LIMIT,
            "remaining": RATE_LIMIT - requests_count - 1,
            "window": RATE_WINDOW
        }


def add_rate_limit_headers(handler, info: dict):
    """Добавить заголовки rate limit в ответ"""
    handler.send_header('X-RateLimit-Limit', str(info['limit']))
    handler.send_header('X-RateLimit-Remaining', str(info['remaining']))
    handler.send_header('X-RateLimit-Window', str(info['window']))
    if 'retry_after' in info:
        handler.send_header('Retry-After', str(info['retry_after']))


def send_rate_limit_error(handler, info: dict):
    """Отправить ответ об ошибке rate limit"""
    import json

    handler.send_response(429)
    handler.send_header('Content-Type', 'application/json')
    handler.send_header('Access-Control-Allow-Origin', '*')
    add_rate_limit_headers(handler, info)
    handler.end_headers()

    handler.wfile.write(json.dumps({
        "error": f"Слишком много запросов. Подождите {info['retry_after']} секунд.",
        "retry_after": info['retry_after']
    }, ensure_ascii=False).encode('utf-8'))
