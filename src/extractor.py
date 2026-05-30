"""PDF extraction module for structured data extraction from resumes."""

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import openai
    from openai import OpenAIError
except ImportError:
    openai = None
    OpenAIError = Exception

try:
    import requests
except ImportError:
    requests = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_path
except ImportError:
    convert_from_path = None


class Extractor:
    """Extract structured data from resumes using PDF parsing and LLMs."""

    def __init__(self, config, logger=None):
        """Initialize extractor with configuration."""
        self.config = config
        self.logger = logger
        self.api_call_count = 0
        self.total_latency = 0.0

    def extract(self, resume_text: str, prompt: str = None) -> Dict:
        """Extract structured data from resume text using an LLM."""
        if prompt is None:
            prompt = self.config.optimizer.baseline_prompt

        formatted_prompt = prompt.replace("{resume_text}", resume_text)

        start_time = time.time()
        llm_result = self._call_llm(formatted_prompt)
        latency = time.time() - start_time

        self.api_call_count += 1
        self.total_latency += latency

        extracted = self._normalize_schema(llm_result["extracted"])
        token_usage = llm_result.get("token_usage", {})
        model_name = llm_result.get("model_name", self.config.model.name)

        self._log_api_call(
            model=model_name,
            prompt=formatted_prompt,
            latency=latency,
            token_usage=token_usage,
        )

        return {
            "extracted": extracted,
            "latency": latency,
            "api_calls": 1,
            "prompt_used": prompt,
            "model_name": model_name,
            "token_usage": token_usage,
        }

    def extract_from_pdf(self, pdf_path: str, prompt: str = None) -> Dict:
        """Extract structured data from a PDF resume file."""
        resume_text = self._extract_text_from_pdf(pdf_path)
        return self.extract(resume_text, prompt)

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        if fitz is None:
            raise ImportError("PyMuPDF is required for PDF extraction. Install PyMuPDF in requirements.txt.")
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        errors = []
        try:
            doc = fitz.open(pdf_path)
            pages = [page.get_text() for page in doc]
            doc.close()
            resume_text = "\n".join(pages).strip()
            if resume_text:
                self._log_text_extraction(pdf_path, "pymupdf")
                return resume_text
            errors.append("PyMuPDF extracted no text")
            self._log_error("PyMuPDF extracted no text", RuntimeError("No text extracted"), {"stage": "pymupdf", "pdf_path": pdf_path})
        except Exception as exc:
            errors.append(f"PyMuPDF error: {exc}")
            self._log_error("PyMuPDF extraction failed", exc, {"stage": "pymupdf", "pdf_path": pdf_path})

        if pdfplumber is not None:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages = [page.extract_text() or "" for page in pdf.pages]
                resume_text = "\n".join(pages).strip()
                if resume_text:
                    self._log_text_extraction(pdf_path, "pdfplumber")
                    return resume_text
                errors.append("pdfplumber extracted no text")
                self._log_error("pdfplumber extracted no text", RuntimeError("No text extracted"), {"stage": "pdfplumber", "pdf_path": pdf_path})
            except Exception as exc:
                errors.append(f"pdfplumber error: {exc}")
                self._log_error("pdfplumber extraction failed", exc, {"stage": "pdfplumber", "pdf_path": pdf_path})

        try:
            return self._extract_text_with_ocr(pdf_path)
        except Exception as exc:
            errors.append(f"OCR error: {exc}")
            self._log_error("OCR extraction failed", exc, {"stage": "ocr", "pdf_path": pdf_path})

        error_msg = "; ".join(errors)
        raise RuntimeError(f"Failed to extract text from PDF {pdf_path}: {error_msg}")

    def _extract_text_with_ocr(self, pdf_path: str) -> str:
        if convert_from_path is None or pytesseract is None:
            raise ImportError("OCR requires pytesseract and pdf2image. Install them in requirements.txt.")

        pages = convert_from_path(pdf_path, dpi=300)
        if not pages:
            raise RuntimeError("OCR could not convert PDF pages to images.")

        extracted_pages = []
        for page_number, page_image in enumerate(pages, start=1):
            try:
                text = pytesseract.image_to_string(page_image)
            except Exception as exc:
                raise RuntimeError(f"OCR failed on page {page_number}: {exc}") from exc
            extracted_pages.append(text or "")

        resume_text = "\n".join(extracted_pages).strip()
        if not resume_text:
            raise ValueError("OCR extracted no text from PDF.")

        self._log_text_extraction(pdf_path, "ocr", len(extracted_pages))
        return resume_text

    def _log_text_extraction(self, pdf_path: str, method: str, ocr_pages: int = 0):
        message = f"Extracted text from {pdf_path} using {method}"
        if method == "ocr":
            message += f" on {ocr_pages} page(s)"
        if self.logger and hasattr(self.logger, "log_event"):
            self.logger.log_event("INFO", message, {"method": method, "pdf_path": pdf_path, "ocr_pages": ocr_pages})
        elif self.config.logging.log_to_console:
            print(f"[Extractor] {message}")

    def _log_error(self, message: str, exc: Exception, extra_data: Optional[Dict] = None):
        details = {"exception": repr(exc), "error_type": type(exc).__name__}
        if extra_data:
            details.update(extra_data)

        if self.logger and hasattr(self.logger, "log_event"):
            self.logger.log_event("ERROR", message, details)
        elif self.config.logging.log_to_console:
            print(f"[ERROR] {message}: {repr(exc)}")

    def _call_llm(self, prompt: str) -> Dict:
        provider = self._resolve_provider()
        if provider == "openai":
            return self._call_openai(prompt)
        if provider == "gemini":
            return self._call_gemini(prompt)
        raise ValueError(f"Unsupported LLM provider: {provider}")

    def _resolve_provider(self) -> str:
        provider = getattr(self.config.model, "provider", "").strip().lower()
        if provider:
            return provider

        model_name = self.config.model.name.lower()
        if "gemini" in model_name or "bison" in model_name:
            return "gemini"
        return "openai"

    def _call_openai(self, prompt: str) -> Dict:
        if openai is None:
            raise ImportError("OpenAI package is not installed. Install openai in requirements.txt.")
        if not self.config.model.api_key:
            raise ValueError("OpenAI API key is required for OpenAI model calls.")

        openai.api_key = self.config.model.api_key
        payload = {
            "model": self.config.model.name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.model.temperature,
            "max_tokens": self.config.model.max_tokens,
        }

        response = self._request_with_retries(lambda: openai.ChatCompletion.create(**payload))
        content = self._get_openai_response_text(response)
        extracted = self._parse_json_from_text(content)
        usage = getattr(response, "usage", None)
        token_usage = self._extract_openai_usage(usage)

        return {"extracted": extracted, "model_name": self.config.model.name, "token_usage": token_usage}

    def _get_openai_response_text(self, response: Any) -> str:
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message"):
                return getattr(choice.message, "content", "") or choice.message["content"]
            if isinstance(choice, dict) and "message" in choice:
                return choice["message"]["content"]

        if isinstance(response, dict):
            return response["choices"][0]["message"]["content"]

        raise ValueError("OpenAI response did not contain expected chat completion content.")

    def _extract_openai_usage(self, usage: Any) -> Dict[str, Optional[int]]:
        if not usage:
            return {}
        if isinstance(usage, dict):
            return {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            }
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    def _call_gemini(self, prompt: str) -> Dict:
        if requests is None:
            raise ImportError("requests is required for Gemini API support. Install requests in requirements.txt.")
        if not self.config.model.api_key:
            raise ValueError("Gemini API key is required for Gemini model calls.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model.name}:generateContent?key={self.config.model.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.config.model.temperature,
                "maxOutputTokens": self.config.model.max_tokens,
            },
        }

        response = self._request_with_retries(
            lambda: requests.post(url, headers=headers, json=payload, timeout=30)
        )
        response.raise_for_status()

        parsed = response.json()
        candidates = parsed.get("candidates") or []
        if not candidates or not isinstance(candidates[0].get("content"), dict):
            raise ValueError("Gemini response did not contain expected content structure.")

        content = candidates[0]["content"]
        parts = content.get("parts") or []
        if not parts or not isinstance(parts[0].get("text"), str):
            raise ValueError("Gemini response did not contain expected text part.")

        text_response = parts[0]["text"]
        print("\n===== GEMINI RAW RESPONSE =====")
        print(text_response)
        print("===== END RESPONSE =====\n")

        extracted = self._parse_json_from_text(text_response)
        token_usage = parsed.get("usageMetadata", {})

        return {"extracted": extracted, "model_name": self.config.model.name, "token_usage": token_usage}

    def _request_with_retries(self, fn: Callable[[], Any]) -> Any:
        attempts = getattr(self.config.model, "retry_attempts", 3)
        backoff = getattr(self.config.model, "retry_backoff_seconds", 1.0)
        last_exception = None

        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except Exception as exc:
                last_exception = exc
                self._log_error(
                    f"LLM request attempt {attempt} failed",
                    exc,
                    {"attempt": attempt, "max_attempts": attempts}
                )
                if attempt == attempts:
                    raise RuntimeError(f"LLM request failed after {attempts} attempts: {exc}") from exc
                time.sleep(backoff * (2 ** (attempt - 1)))

        raise last_exception

    def _parse_json_from_text(self, text: str) -> Dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("LLM response text is empty.")

        text = text.strip()
        decoder = json.JSONDecoder()
        index = 0
        while index < len(text):
            try:
                parsed_obj, _ = decoder.raw_decode(text[index:])
                if isinstance(parsed_obj, list) and len(parsed_obj) == 1 and isinstance(parsed_obj[0], dict):
                    parsed_obj = parsed_obj[0]
                if not isinstance(parsed_obj, dict):
                    raise ValueError("Expected JSON object from LLM response.")
                return parsed_obj
            except json.JSONDecodeError:
                index += 1

        raise ValueError("Could not parse JSON from LLM response.")

    def _normalize_schema(self, extracted: Any) -> Dict:
        defaults = {
            "name": "",
            "email": "",
            "phone": "",
            "skills": [],
            "experience": [],
            "education": [],
            "certifications": [],
        }

        normalized = {}
        if not isinstance(extracted, dict):
            extracted = {}

        for field in self.config.schema.fields:
            value = extracted.get(field, defaults[field])
            if field in {"skills", "experience", "education", "certifications"}:
                normalized[field] = self._normalize_list_field(value)
            else:
                normalized[field] = value if value is not None else defaults[field]

        return normalized

    def _normalize_list_field(self, value: Any) -> List:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [item.strip() for item in re.split(r"[\n,;]+", value) if item.strip()]
        return [value]

    def extract_batch(self, resume_texts: List[str], prompt: str = None) -> List[Dict]:
        """Extract from multiple resumes."""
        return [self.extract(text, prompt) for text in resume_texts]

    def get_stats(self) -> Dict:
        """Get API call statistics."""
        avg_latency = self.total_latency / self.api_call_count if self.api_call_count > 0 else 0.0
        return {
            "total_api_calls": self.api_call_count,
            "total_latency": self.total_latency,
            "average_latency": avg_latency,
        }

    def _log_api_call(
        self,
        model: str,
        prompt: str,
        latency: float,
        token_usage: Optional[Dict[str, Any]] = None,
    ):
        if self.logger:
            tokens = 0
            if token_usage:
                tokens = sum(int(token_usage.get(k, 0) or 0) for k in ["total_tokens", "prompt_tokens", "completion_tokens"])
            if hasattr(self.logger, "log_api_call"):
                self.logger.log_api_call(model=model, tokens_used=tokens, cost=0.0)
            if hasattr(self.logger, "log_latency"):
                self.logger.log_latency("extract", latency)
        elif self.config.logging.log_to_console:
            print(f"[Extractor] model={model} latency={latency:.3f}s prompt_length={len(prompt)} tokens={token_usage}")
