from http.server import BaseHTTPRequestHandler
import json
import os
import re
import urllib.request
from api.rate_limiter import get_client_ip, check_rate_limit, send_rate_limit_error, add_rate_limit_headers

# API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Gemini setup
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

# Веса документов по типу (для приоритезации источников)
DOCUMENT_WEIGHTS = {
    "plenum_resolution": 1.5,    # Постановления Пленума ВС РФ - высший приоритет
    "practice_review": 1.3,      # Обзоры судебной практики
    "court_decision": 1.0,       # Судебные решения - базовый вес
    "scientific_article": 0.8,   # Научные статьи - справочно
    "ai_review": 0.6             # AI обзоры - низший приоритет
}

# Синонимы для расширения запроса (Query Expansion)
LEGAL_SYNONYMS = {
    "неустойка": ["пеня", "штраф", "санкция", "штрафная санкция"],
    "снижение": ["уменьшение", "редукция", "снизить", "уменьшить"],
    "несоразмерность": ["чрезмерность", "явная несоразмерность", "несоразмерный"],
    "договор поставки": ["поставка товара", "поставщик", "покупатель"],
    "договор подряда": ["подрядчик", "заказчик", "подрядные работы"],
    "договор аренды": ["арендатор", "арендодатель", "арендная плата"],
    "просрочка": ["нарушение срока", "несвоевременно", "задержка"],
    "ответчик": ["должник", "нарушитель"],
    "истец": ["кредитор", "взыскатель"],
}


def expand_query(query: str) -> str:
    """
    Расширяет поисковый запрос юридическими синонимами.
    Улучшает recall при векторном поиске.
    """
    query_lower = query.lower()
    expansions = []

    for term, synonyms in LEGAL_SYNONYMS.items():
        if term in query_lower:
            # Добавляем синонимы, которых ещё нет в запросе
            for syn in synonyms:
                if syn.lower() not in query_lower:
                    expansions.append(syn)

    if expansions:
        # Добавляем уникальные синонимы в конец запроса
        return query + " " + " ".join(expansions[:5])  # Максимум 5 синонимов

    return query


def calculate_temporal_boost(decision_date: str) -> float:
    """
    Вычисляет бонус за свежесть документа.
    Более новые решения получают больший вес.
    """
    if not decision_date:
        return 1.0

    try:
        from datetime import datetime
        # Парсим дату (формат YYYY-MM-DD)
        doc_date = datetime.strptime(decision_date[:10], "%Y-%m-%d")
        current_year = datetime.now().year

        year_diff = current_year - doc_date.year

        if year_diff <= 1:      # 2024-2025
            return 1.2
        elif year_diff <= 3:    # 2022-2023
            return 1.1
        elif year_diff <= 5:    # 2020-2021
            return 1.0
        else:                   # старше 5 лет
            return 0.9
    except (ValueError, TypeError):
        return 1.0


def calculate_outcome_boost(penalty_reduced: bool, category: str) -> float:
    """
    Вычисляет бонус на основе исхода дела.
    Для истца ценнее дела, где неустойка НЕ была снижена.
    """
    # Для постановлений и обзоров исход не применим
    if category in ['plenum_resolution', 'practice_review', 'scientific_article', 'ai_review']:
        return 1.0

    if penalty_reduced is None:
        return 1.0

    if penalty_reduced:
        # Неустойка снижена - менее ценно для истца, но полезно знать аргументы
        return 0.9
    else:
        # Неустойка НЕ снижена - очень ценно для истца
        return 1.3


# Claude setup - lazy import
anthropic_client = None

def get_anthropic_client():
    global anthropic_client
    if anthropic_client is None and ANTHROPIC_API_KEY:
        import anthropic
        anthropic_client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            timeout=290.0  # 290 сек для думающих моделей (в пределах 300 сек Vercel Pro)
        )
    return anthropic_client


# OpenAI setup - lazy import
openai_client = None

def get_openai_client():
    global openai_client
    if openai_client is None and OPENAI_API_KEY:
        from openai import OpenAI
        openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=290.0  # 290 сек для думающих моделей (o1, o3)
        )
    return openai_client


def extract_relevant_excerpt(full_text: str, query: str, max_length: int = 500) -> str:
    """
    Извлекает наиболее релевантный фрагмент из полного текста документа.
    Ищет фрагменты с ключевыми юридическими терминами.
    """
    if not full_text or len(full_text) <= max_length:
        return full_text or ""

    # Ключевые слова для поиска релевантных фрагментов (в порядке приоритета)
    priority_keywords = [
        "333", "несоразмерн", "явн", "чрезмерн",  # ст. 333 ГК
        "снижен", "уменьшен", "редуц",            # снижение неустойки
        "неустойк", "пен", "штраф",               # виды санкций
        "установил", "пришёл к выводу", "указал", # выводы суда
        "доказательств", "бремя",                 # доказывание
        "ответчик", "истец"                       # стороны
    ]

    best_excerpt = ""
    best_score = 0

    # Разбиваем на абзацы
    paragraphs = full_text.split('\n\n')
    if len(paragraphs) == 1:
        paragraphs = full_text.split('\n')

    for para in paragraphs:
        para = para.strip()
        if len(para) < 50:  # Пропускаем короткие абзацы
            continue

        # Считаем релевантность абзаца
        score = 0
        para_lower = para.lower()
        for i, kw in enumerate(priority_keywords):
            if kw in para_lower:
                score += (len(priority_keywords) - i)  # Приоритетные слова дают больше очков

        if score > best_score:
            best_score = score
            best_excerpt = para[:max_length]
            if len(para) > max_length:
                # Обрезаем по границе предложения
                last_period = best_excerpt.rfind('.')
                if last_period > max_length // 2:
                    best_excerpt = best_excerpt[:last_period + 1]

    # Если не нашли релевантный фрагмент, берём начало
    if not best_excerpt:
        best_excerpt = full_text[:max_length]
        last_period = best_excerpt.rfind('.')
        if last_period > max_length // 2:
            best_excerpt = best_excerpt[:last_period + 1]

    return best_excerpt


def calculate_keyword_score(text: str, query: str) -> float:
    """
    Вычисляет keyword score на основе совпадения ключевых слов.
    Возвращает значение от 0 до 1.
    """
    if not text or not query:
        return 0.0

    text_lower = text.lower()
    query_lower = query.lower()

    # Извлекаем слова из запроса (минимум 3 символа)
    query_words = [w for w in re.findall(r'\b\w+\b', query_lower) if len(w) >= 3]
    if not query_words:
        return 0.0

    # Считаем совпадения
    matches = 0
    for word in query_words:
        if word in text_lower:
            matches += 1

    # Бонус за точные фразы (2+ слова подряд)
    phrase_bonus = 0
    for i in range(len(query_words) - 1):
        phrase = query_words[i] + " " + query_words[i + 1]
        if phrase in text_lower:
            phrase_bonus += 0.2

    base_score = matches / len(query_words) if query_words else 0
    return min(1.0, base_score + phrase_bonus)


def hybrid_rerank(results: list, query: str, vector_weight: float = 0.7, keyword_weight: float = 0.3) -> list:
    """
    Гибридное переранжирование: комбинация векторного сходства, keyword matching,
    временного бонуса и бонуса за исход дела.

    Args:
        results: Результаты векторного поиска
        query: Исходный поисковый запрос
        vector_weight: Вес векторного скора (по умолчанию 0.7)
        keyword_weight: Вес keyword скора (по умолчанию 0.3)

    Returns:
        Переранжированный список результатов
    """
    if not results:
        return results

    for item in results:
        # 1. Векторный скор (уже есть от Supabase)
        vector_score = item.get('similarity', 0.5)

        # 2. Keyword скор (вычисляем по summary + full_text)
        searchable_text = (item.get('summary', '') + ' ' + item.get('full_text', '')[:2000])
        keyword_score = calculate_keyword_score(searchable_text, query)

        # 3. Вес типа документа
        category = item.get('category', 'court_decision')
        type_weight = DOCUMENT_WEIGHTS.get(category, 1.0)

        # 4. Temporal boost (свежие решения важнее)
        decision_date = item.get('decision_date', '')
        temporal_boost = calculate_temporal_boost(decision_date)

        # 5. Outcome boost (дела без снижения ценнее для истца)
        penalty_reduced = item.get('penalty_reduced')
        outcome_boost = calculate_outcome_boost(penalty_reduced, category)

        # 6. Финальный гибридный скор с учётом всех факторов
        base_score = vector_weight * vector_score + keyword_weight * keyword_score
        hybrid_score = base_score * type_weight * temporal_boost * outcome_boost

        # Сохраняем все скоры для отладки
        item['vector_score'] = vector_score
        item['keyword_score'] = keyword_score
        item['temporal_boost'] = temporal_boost
        item['outcome_boost'] = outcome_boost
        item['hybrid_score'] = hybrid_score

    # Сортируем по гибридному скору
    results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
    return results


def rerank_by_document_type(results: list) -> list:
    """Переранжирование результатов с учётом веса типа документа"""
    if not results:
        return results

    for item in results:
        category = item.get('category', 'court_decision')
        weight = DOCUMENT_WEIGHTS.get(category, 1.0)
        similarity = item.get('similarity', 0.5)
        # Взвешенный скор = similarity * weight
        item['weighted_score'] = similarity * weight

    # Сортируем по взвешенному скору (убывание)
    results.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
    return results


def compress_text(text: str) -> str:
    """Сжатие текста для экономии токенов"""
    if not text:
        return ''
    # Нормализация переносов строк
    text = text.replace('\r\n', '\n')
    # Удаление множественных пустых строк
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Удаление множественных пробелов
    text = re.sub(r'[ \t]{2,}', ' ', text)
    # Удаление пробелов в начале и конце строк
    text = re.sub(r'^[ \t]+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text.strip()


def search_court_decisions(query_text: str) -> list:
    """Поиск похожих судебных решений в базе знаний"""
    if not SUPABASE_URL or not SUPABASE_KEY or not OPENAI_API_KEY:
        return []

    try:
        # Получаем embedding для запроса
        text = query_text[:30000]
        embed_data = json.dumps({
            "input": text,
            "model": "text-embedding-3-small"
        }).encode('utf-8')

        embed_req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=embed_data,
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
        )

        with urllib.request.urlopen(embed_req, timeout=15) as response:
            embed_result = json.loads(response.read().decode('utf-8'))
            query_embedding = embed_result['data'][0]['embedding']

        # Поиск в Supabase (понижен порог для лучшего recall)
        search_data = json.dumps({
            "query_embedding": query_embedding,
            "match_count": 7,  # Берём больше, потом отфильтруем переранжированием
            "match_threshold": 0.4  # Понижен с 0.6 для лучшего recall
        }).encode('utf-8')

        search_req = urllib.request.Request(
            f"{SUPABASE_URL}/rest/v1/rpc/search_decisions",
            data=search_data,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(search_req, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print(f"Court decisions search error: {e}")
        return []


def build_system_prompt(rates_info: str, court_cases: list = None) -> str:
    # Формируем блок с релевантными источниками
    court_practice = ""
    if court_cases:
        court_practice = "\n\nРЕЛЕВАНТНЫЕ ИСТОЧНИКИ ИЗ БАЗЫ ЗНАНИЙ (используй в аргументации):\n"
        for i, case in enumerate(court_cases, 1):
            category = case.get('category', 'court_decision')
            case_number = case.get('case_number', '')
            court_name = case.get('court_name', '')
            summary = case.get('summary', '')[:400]
            key_points = case.get('key_points', [])

            # Определяем тип документа для отображения
            is_court_decision = False
            has_number = False

            if category == 'plenum_resolution' or 'пленум' in category.lower() or 'пленум' in summary.lower():
                doc_label = "Постановление Пленума"
                has_number = True
            elif category == 'practice_review' or 'обзор' in category.lower():
                doc_label = "Обзор практики"
                has_number = True
            elif category == 'scientific_article' or 'статья' in category.lower() or 'научн' in category.lower():
                doc_label = "Научная статья"
            elif category == 'ai_review':
                doc_label = "Аналитический обзор"
            else:
                doc_label = "Дело"
                is_court_decision = True
                has_number = True

            court_practice += f"\n{i}. {doc_label}"
            if has_number and case_number:
                court_practice += f" {case_number}"
            if court_name:
                court_practice += f" ({court_name})"

            # Добавляем дату решения
            decision_date = case.get('decision_date', '')
            if decision_date:
                court_practice += f" от {decision_date[:10]}"

            court_practice += f"\n   {summary}\n"

            # Контекстное обогащение: ставка и сумма неустойки (если есть)
            penalty_rate = case.get('penalty_rate', '')
            penalty_amount = case.get('penalty_amount', '')
            if penalty_rate or penalty_amount:
                context_parts = []
                if penalty_rate:
                    context_parts.append(f"ставка: {penalty_rate}")
                if penalty_amount:
                    context_parts.append(f"сумма: {penalty_amount}")
                court_practice += f"   Параметры неустойки: {', '.join(context_parts)}\n"

            if key_points:
                court_practice += f"   Ключевые позиции: {', '.join(key_points[:4])}\n"

            # Извлекаем релевантную цитату из full_text (если есть)
            full_text = case.get('full_text', '')
            if full_text and len(full_text) > 100:
                excerpt = extract_relevant_excerpt(full_text, "", max_length=400)
                if excerpt and excerpt != summary[:400]:
                    court_practice += f"   Цитата: «{excerpt}»\n"

            # Только для судебных решений показываем исход по неустойке
            if is_court_decision and case.get('penalty_reduced') is not None:
                reduced = "снижена" if case.get('penalty_reduced') else "не снижена"
                percent = f" на {case.get('reduction_percent')}%" if case.get('reduction_percent') else ""
                court_practice += f"   Исход: неустойка {reduced}{percent}\n"
    return f"""Ты — опытный российский юрист. Твоя задача — создать профессиональный аналитический документ в стиле Кузнецова.

СТИЛЬ КУЗНЕЦОВА — принципы:
- Формальный деловой язык без разговорных оборотов
- Безличные конструкции: «представляется обоснованным», «необходимо отметить», «следует учитывать»
- Нумерованные разделы с иерархией (1., 2., 2.1., 2.2.)
- Точные ссылки на нормы права: «п. 1 ст. 333 ГК РФ», «п. 71 Постановления Пленума ВС РФ от 24.03.2016 № 7»
- Конкретные даты (формат: 01.01.2025), суммы, проценты из документов
- Каждый тезис подкреплён фактами из дела

СТРУКТУРА ДОКУМЕНТА:

1. ФАКТИЧЕСКИЕ ОБСТОЯТЕЛЬСТВА ДЕЛА
   Краткое изложение: стороны, договор, сумма долга, период просрочки, размер неустойки.

2. ПРАВОВОЕ ОБОСНОВАНИЕ НЕДОПУСТИМОСТИ СНИЖЕНИЯ НЕУСТОЙКИ
   2.1. [Первый аргумент — заголовок]
   Текст аргумента с конкретными ссылками на факты дела и нормы права.

   2.2. [Второй аргумент — заголовок]
   ...и так далее (5-7 аргументов)

3. АНАЛИЗ ДОВОДОВ ОТВЕТЧИКА (если есть отзыв)
   3.1. Довод ответчика о [суть довода]
   Контраргумент с обоснованием.

4. ЗАКЛЮЧЕНИЕ
   Краткий вывод: оснований для применения ст. 333 ГК РФ не имеется.

5. ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ
   Перечисли все источники из базы знаний, на которые ты ссылался в тексте.
   Формат: номер дела или название документа, суд/орган, дата.

ОБЯЗАТЕЛЬНО ИСПОЛЬЗУЙ ДАННЫЕ ИЗ ДОКУМЕНТОВ:
- Наименования сторон (ООО, АО, ИП — как в документах)
- Номер и дату договора
- Конкретные суммы (основной долг, неустойка)
- Период просрочки с датами
- Процентную ставку неустойки по договору
- Доводы ответчика (если есть отзыв) — каждый довод разбери отдельно

ПРАВИЛА ЦИТИРОВАНИЯ СУДЕБНОЙ ПРАКТИКИ:
- При ссылке на судебное решение или постановление из базы знаний ОБЯЗАТЕЛЬНО приводи краткую цитату
- Цитаты оформляй курсивом: _«текст цитаты»_
- ВАЖНО: Цитата должна начинаться С НОВОГО АБЗАЦА (новой строки)
- Формат цитирования:
  [Твой текст со ссылкой на дело]

  _«Цитата из решения суда»_

  [Продолжение твоего текста]
- Пример:
  Арбитражный суд в деле А40-12345/2024 подтвердил данную позицию:

  _«Неустойка в размере 0,1% в день является обычно применяемой в деловом обороте и не может быть признана чрезмерной»_

- Цитируй ключевые выводы суда, подтверждающие твою позицию

АКТУАЛЬНЫЕ СТАВКИ ЦБ РФ:
{rates_info}
{court_practice}
ФОРМАТ ВЫВОДА:
- Текст без markdown, КРОМЕ курсива для цитат (используй _текст_ для курсива)
- Нумерация разделов арабскими цифрами с точкой
- Абзацы разделены пустой строкой
- Профессиональный юридический язык
- Обязательно ссылайся на релевантную судебную практику с цитатами
- В конце обязательно добавь раздел "5. ИСПОЛЬЗОВАННЫЕ ИСТОЧНИКИ" со списком всех использованных документов из базы знаний"""


def build_user_prompt(claim_text: str, response_text: str, other_docs: str, comments: str) -> str:
    prompt_parts = []

    prompt_parts.append("ИСХОДНЫЕ МАТЕРИАЛЫ ДЛЯ АНАЛИЗА")
    prompt_parts.append("")
    prompt_parts.append("Приложение 1. Исковое заявление о взыскании неустойки")
    prompt_parts.append("-" * 50)
    prompt_parts.append(claim_text)

    if response_text and response_text.strip():
        prompt_parts.append("")
        prompt_parts.append("Приложение 2. Отзыв ответчика / Ходатайство о применении ст. 333 ГК РФ")
        prompt_parts.append("-" * 50)
        prompt_parts.append(response_text)

    if other_docs and other_docs.strip():
        prompt_parts.append("")
        prompt_parts.append("Приложение 3. Дополнительные материалы")
        prompt_parts.append("-" * 50)
        prompt_parts.append(other_docs)

    if comments and comments.strip():
        prompt_parts.append("")
        prompt_parts.append("Указания от пользователя:")
        prompt_parts.append(comments)

    prompt_parts.append("")
    prompt_parts.append("=" * 50)
    prompt_parts.append("ЗАДАНИЕ: Составь аналитический документ в стиле Кузнецова.")
    prompt_parts.append("Извлеки из материалов все факты и создай структурированное правовое обоснование.")

    return "\n".join(prompt_parts)


def clean_markdown(text: str) -> str:
    """Убираем markdown из текста, сохраняя курсив (_текст_)"""
    # Убираем жирный текст и заголовки, но оставляем курсив (подчёркивания)
    text = text.replace('**', '')
    text = text.replace('###', '')
    text = text.replace('##', '')
    # Убираем одиночные * но не трогаем _
    # Заменяем *текст* на _текст_ для единообразия курсива
    import re
    text = re.sub(r'\*([^*]+)\*', r'_\1_', text)
    return text


def call_gemini(system_prompt: str, user_prompt: str) -> str:
    """Вызов Gemini API"""
    model = genai.GenerativeModel(
        model_name="gemini-3-pro-preview",
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

    # Получаем текст
    try:
        text = response.text
    except (ValueError, AttributeError):
        try:
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        text = candidate.content.parts[0].text
        except (AttributeError, IndexError):
            raise Exception("Не удалось получить ответ от Gemini")

    return clean_markdown(text)


def call_claude(system_prompt: str, user_prompt: str, model_id: str) -> str:
    """Вызов Claude API"""
    client = get_anthropic_client()
    if not client:
        raise Exception("Claude API недоступен. Проверьте ANTHROPIC_API_KEY")

    try:
        message = client.messages.create(
            model=model_id,
            max_tokens=8192,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )
        text = message.content[0].text
        return clean_markdown(text)
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            raise Exception("Claude API не ответил вовремя. Попробуйте Gemini или повторите позже.")
        elif "overloaded" in error_msg:
            raise Exception("Claude API перегружен. Попробуйте позже или используйте Gemini.")
        elif "rate" in error_msg:
            raise Exception("Превышен лимит запросов Claude. Подождите минуту.")
        elif "invalid" in error_msg and "key" in error_msg:
            raise Exception("Неверный API ключ Claude.")
        elif "connection" in error_msg:
            raise Exception("Не удалось подключиться к Claude API. Проверьте интернет.")
        else:
            raise Exception(f"Ошибка Claude: {str(e)[:200]}")


def call_openai(system_prompt: str, user_prompt: str, model_id: str) -> str:
    """Вызов OpenAI API"""
    client = get_openai_client()
    if not client:
        raise Exception("OpenAI API недоступен. Проверьте OPENAI_API_KEY")

    try:
        # Только o1/o3 - настоящие reasoning модели, требуют developer role
        is_reasoning_model = any(x in model_id for x in ['o1', 'o3'])

        if is_reasoning_model:
            # Для o1/o3 используем developer role
            messages = [
                {"role": "developer", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat.completions.create(
                model=model_id,
                max_completion_tokens=8192,
                messages=messages
            )
        else:
            # Для gpt-4, gpt-5 и других - стандартный формат
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            response = client.chat.completions.create(
                model=model_id,
                max_completion_tokens=8192,
                messages=messages
            )

        # Получаем текст ответа
        text = None
        message = response.choices[0].message

        # Пробуем разные способы получить текст
        if message.content:
            text = message.content
        elif hasattr(message, 'reasoning_content') and message.reasoning_content:
            text = message.reasoning_content
        elif hasattr(message, 'refusal') and message.refusal:
            raise Exception(f"Модель отказалась отвечать: {message.refusal[:100]}")

        if not text:
            # Для отладки - показываем структуру ответа
            msg_dict = message.model_dump() if hasattr(message, 'model_dump') else str(message)
            raise Exception(f"Пустой ответ. Структура: {str(msg_dict)[:150]}")

        return clean_markdown(text)
    except Exception as e:
        error_msg = str(e).lower()
        if "timeout" in error_msg or "timed out" in error_msg:
            raise Exception("OpenAI API не ответил вовремя. Попробуйте другую модель.")
        elif "rate" in error_msg:
            raise Exception("Превышен лимит запросов OpenAI. Подождите минуту.")
        elif "invalid" in error_msg and "key" in error_msg:
            raise Exception("Неверный API ключ OpenAI.")
        else:
            raise Exception(f"Ошибка OpenAI: {str(e)[:200]}")


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Rate limiting: 3 запроса в минуту с одного IP
            client_ip = get_client_ip(self.headers)
            allowed, rate_info = check_rate_limit(client_ip)
            if not allowed:
                send_rate_limit_error(self, rate_info)
                return

            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            claim_text = compress_text(data.get('claim_text', ''))
            response_text = compress_text(data.get('response_text', ''))
            other_documents = compress_text(data.get('other_documents', ''))
            user_comments = data.get('user_comments', '').strip()
            rates_info = data.get('rates_info', 'Ставки ЦБ недоступны')
            model = data.get('model', 'gemini-3-pro-preview')

            if not claim_text:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Текст иска обязателен для анализа"
                }).encode())
                return

            # Проверяем доступность API
            is_claude = model.startswith('claude')
            is_openai = model.startswith('gpt') or model.startswith('o1') or model.startswith('o3')
            is_gemini = not is_claude and not is_openai

            if is_claude and not ANTHROPIC_API_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Claude API не настроен. Добавьте ANTHROPIC_API_KEY в Vercel"
                }).encode())
                return

            if is_openai and not OPENAI_API_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "OpenAI API не настроен. Добавьте OPENAI_API_KEY в Vercel"
                }).encode())
                return

            if is_gemini and not GEMINI_API_KEY:
                self.send_response(503)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Gemini API не настроен. Добавьте GEMINI_API_KEY в Vercel"
                }).encode())
                return

            # Поиск похожих судебных решений в базе знаний (RAG)
            court_cases = []
            rag_status = "not_attempted"
            rag_error = None
            try:
                search_query = claim_text[:5000]  # Используем начало иска для поиска
                # Query Expansion: расширяем запрос юридическими синонимами
                expanded_query = expand_query(search_query)
                court_cases = search_court_decisions(expanded_query)
                # Гибридное переранжирование: vector + keyword + temporal + outcome
                if court_cases:
                    court_cases = hybrid_rerank(court_cases, search_query)
                    rag_status = "success"
                    print(f"RAG: found {len(court_cases)} documents")
                else:
                    rag_status = "no_matches"
                    print("RAG: no matching documents found")
            except Exception as e:
                rag_status = "error"
                rag_error = str(e)[:100]
                print(f"RAG search failed: {e}")  # Продолжаем без RAG если поиск не удался

            system_prompt = build_system_prompt(rates_info, court_cases)
            user_prompt = build_user_prompt(
                claim_text,
                response_text,
                other_documents,
                user_comments
            )

            # Вызываем нужную модель
            if is_claude:
                arguments_text = call_claude(system_prompt, user_prompt, model)
            elif is_openai:
                arguments_text = call_openai(system_prompt, user_prompt, model)
            else:
                arguments_text = call_gemini(system_prompt, user_prompt)

            # Формируем информацию об использованных источниках
            sources_used = []
            if court_cases:
                for case in court_cases[:7]:  # Все 7 источников
                    source_info = {
                        "case_number": case.get('case_number', ''),
                        "court_name": case.get('court_name', ''),
                        "category": case.get('category', 'court_decision'),
                        "similarity": round(case.get('similarity', 0), 3),
                        "hybrid_score": round(case.get('hybrid_score', 0), 3)
                    }
                    sources_used.append(source_info)

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            add_rate_limit_headers(self, rate_info)
            self.end_headers()
            self.wfile.write(json.dumps({
                "arguments_text": arguments_text,
                "model_used": model,
                "sources_used": sources_used,
                "sources_count": len(sources_used),
                "rag_status": rag_status,
                "rag_error": rag_error
            }).encode())

        except Exception as e:
            error_msg = str(e)
            # Убираем спецсимволы которые могут сломать JSON
            error_msg = error_msg.replace('\n', ' ').replace('\r', ' ')[:500]
            try:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": f"Ошибка AI: {error_msg}"
                }, ensure_ascii=False).encode('utf-8'))
            except Exception:
                # Fallback если что-то совсем пошло не так
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(b'{"error": "Internal server error"}')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
