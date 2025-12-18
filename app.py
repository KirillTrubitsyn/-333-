from __future__ import annotations
import datetime as dt
import os
from typing import List, Optional
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from dateutil.relativedelta import relativedelta
import google.generativeai as genai

from rates_loader import RatesProvider

app = FastAPI(title="Анализатор возражений по ст. 333 ГК РФ", version="2.0.0")

# Конфигурация
RATES_URL = os.getenv("RATES_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

rates = RatesProvider(source_url=RATES_URL)

# Настройка Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Модели данных
class RateStep(BaseModel):
    date_from: str
    key_rate: float

class AnalyzeRequest(BaseModel):
    claim_text: str = Field(..., description="Текст иска о взыскании неустойки")
    response_text: str = Field(default="", description="Текст отзыва оппонента")
    other_documents: str = Field(default="", description="Другие документы")
    user_comments: str = Field(default="", description="Пояснения и комментарии пользователя")

class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "objection_text": "ВОЗРАЖЕНИЯ против применения ст. 333 ГК РФ...",
        "rates_used": [{"date_from": "2024-01-01", "key_rate": 16.0}],
        "analysis_summary": "Краткий анализ документов"
    }})
    objection_text: str
    rates_used: List[RateStep]
    analysis_summary: str


@app.get("/health")
async def health():
    return {
        "ok": True,
        "version": app.version,
        "rates_url_set": bool(RATES_URL),
        "gemini_configured": bool(GEMINI_API_KEY)
    }


@app.get("/rates")
async def get_rates():
    """Получить текущие ставки ЦБ РФ"""
    try:
        steps = await rates.get_steps()
        return [{"date_from": s[0].isoformat(), "key_rate": s[1]} for s in steps]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Не удалось загрузить ставки ЦБ. Ошибка: {e}")


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

    if response_text.strip():
        prompt_parts.append("\n\n=== ОТЗЫВ ОТВЕТЧИКА (ХОДАТАЙСТВО О СНИЖЕНИИ НЕУСТОЙКИ) ===")
        prompt_parts.append(response_text)

    if other_docs.strip():
        prompt_parts.append("\n\n=== ДОПОЛНИТЕЛЬНЫЕ ДОКУМЕНТЫ ===")
        prompt_parts.append(other_docs)

    if comments.strip():
        prompt_parts.append("\n\n=== ПОЯСНЕНИЯ И КОММЕНТАРИИ ПОЛЬЗОВАТЕЛЯ ===")
        prompt_parts.append(comments)

    prompt_parts.append("\n\nНа основе этих документов подготовь ВОЗРАЖЕНИЯ против применения ст. 333 ГК РФ и снижения неустойки.")

    return "\n".join(prompt_parts)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_documents(request: AnalyzeRequest):
    """Анализ документов и генерация возражений с помощью Gemini AI"""

    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Gemini API не настроен. Установите переменную окружения GEMINI_API_KEY"
        )

    if not request.claim_text.strip():
        raise HTTPException(status_code=400, detail="Текст иска обязателен для анализа")

    # Получаем ставки ЦБ
    try:
        rate_steps = await rates.get_steps()
        rates_info = "\n".join([
            f"- с {s[0].isoformat()}: {s[1]}% годовых"
            for s in rate_steps[-10:]  # Последние 10 ставок
        ])
        rates_list = [{"date_from": s[0].isoformat(), "key_rate": s[1]} for s in rate_steps[-10:]]
    except Exception:
        rates_info = "Ставки ЦБ недоступны. Используй общеизвестные данные о ключевой ставке."
        rates_list = []

    # Формируем промпты
    system_prompt = build_system_prompt(rates_info)
    user_prompt = build_user_prompt(
        request.claim_text,
        request.response_text,
        request.other_documents,
        request.user_comments
    )

    try:
        # Вызов Gemini API
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
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

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обращении к Gemini AI: {str(e)}"
        )

    return AnalyzeResponse(
        objection_text=objection_text,
        rates_used=rates_list,
        analysis_summary=analysis_summary
    )


@app.post("/analyze-files")
async def analyze_files(
    claim_file: Optional[UploadFile] = File(None),
    claim_text: str = Form(""),
    response_file: Optional[UploadFile] = File(None),
    response_text: str = Form(""),
    other_file: Optional[UploadFile] = File(None),
    other_text: str = Form(""),
    user_comments: str = Form("")
):
    """Анализ загруженных файлов и/или текста"""

    # Читаем содержимое файлов или используем текст
    async def get_content(file: Optional[UploadFile], text: str) -> str:
        if file and file.filename:
            content = await file.read()
            try:
                return content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    return content.decode('cp1251')
                except UnicodeDecodeError:
                    return content.decode('utf-8', errors='ignore')
        return text

    final_claim = await get_content(claim_file, claim_text)
    final_response = await get_content(response_file, response_text)
    final_other = await get_content(other_file, other_text)

    if not final_claim.strip():
        raise HTTPException(status_code=400, detail="Необходимо загрузить иск или ввести его текст")

    # Используем основной эндпоинт для анализа
    request = AnalyzeRequest(
        claim_text=final_claim,
        response_text=final_response,
        other_documents=final_other,
        user_comments=user_comments
    )

    return await analyze_documents(request)


# Serve index.html at root
@app.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
