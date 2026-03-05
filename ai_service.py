"""
ai_service.py
─────────────
OpenAI integration layer.

Responsibilities:
  • Segment original document text into logical question blocks using an LLM.
  • Batch process these segments for verification.
  • Call the OpenAI Chat Completion endpoint with retry + exponential back‑off.
  • Reassemble verified segments in order.
"""

from __future__ import annotations

import logging
import json
from typing import Sequence
import concurrent.futures

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

import config

logger = logging.getLogger(__name__)

# ── OpenAI client (initialised once at module level) ─────────────────────────
_client: OpenAI | None = None

def _get_client() -> OpenAI:
    """Lazy‑initialise the OpenAI client so import‑time errors are avoided."""
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
    return _client

# ── Public API ────────────────────────────────────────────────────────────────

def process_document(text: str, user_verification_prompt: str) -> str:
    """
    Accept the full document text, segment it, verify it in batches,
    and return the fully verified document text.
    """
    if not text.strip():
        return ""

    logger.info("Starting segmentation...")
    raw_segments = segment_document(text)
    logger.info("Extracted %d segments.", len(raw_segments))

    segments = clean_segments(raw_segments)
    logger.info("Cleaned down to %d segments.", len(segments))

    if not segments:
        return text

    verified_segments = process_in_batches(segments, user_verification_prompt)

    # Reassemble
    verified_text = "\n\n".join(seg.get("segment_text", "").strip() for seg in verified_segments)
    return verified_text


# ── Internals: Segmentation ───────────────────────────────────────────────────

SEGMENTATION_SYSTEM_PROMPT = """You are a document segmentation assistant.

Your task is to take the provided document text and split it into logical question segments.
A segment is defined as everything from the start of one question until the next question begins.
This includes: the question stem, options, answer key, solution, explanation, hints, and insights.

CRITICAL RULES:
- IMPORTANT: You MUST return ONLY a JSON array.
- Do NOT add markdown wrappers like ```json if you can avoid it, output raw JSON.
- Split the document into logical question segments.
- Do NOT modify the text natively, extract it exactly.
- Do NOT summarize.
- Do NOT invent content.

Expected output format:
[
 {
  "segment_id": 1,
  "segment_text": "full question block exactly as it appeared"
 },
 {
  "segment_id": 2,
  "segment_text": "full question block exactly as it appeared"
 }
]
"""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def call_openai_json_segmenter(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=config.OPENAI_SEGMENT_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()

def segment_document(text: str) -> list[dict]:
    """Uses LLM to segment the document into JSON."""
    user_prompt = f"Please segment the following document text:\n\n{text}"
    # OpenAI JSON mode requires the prompt to include the word "JSON"
    system_prompt = SEGMENTATION_SYSTEM_PROMPT + "\n\nOutput a JSON object with a 'segments' key containing the array."
    
    try:
        raw_json = call_openai_json_segmenter(system_prompt, user_prompt)
        data = json.loads(raw_json)
        # Handle variations in how the model might return it
        if isinstance(data, dict):
            if "segments" in data:
                return data["segments"]
            # Sometimes it just wraps it in some other key
            for k, v in data.items():
                if isinstance(v, list):
                    return v
            return []
        elif isinstance(data, list):
            return data
        return []
    except Exception:
        logger.exception("Segmentation failed.")
        # Fallback: Treat whole doc as one segment if segmentation completely fails
        return [{"segment_id": 1, "segment_text": text}]

def clean_segments(segments: list[dict]) -> list[dict]:
    """
    Remove empty segments, duplicates, trim whitespace, 
    and discard malformed fragments.
    """
    cleaned = []
    seen = set()
    
    for seg in segments:
        text = seg.get("segment_text", "")
        if not isinstance(text, str):
            continue
            
        text = text.strip()
        if not text:
            continue
            
        if text in seen:
            continue
            
        seen.add(text)
        cleaned.append({
            "segment_id": len(cleaned) + 1,  # Reassign sequential IDs
            "segment_text": text
        })
        
    return cleaned

# ── Internals: Batch Processing & Verification ─────────────────────────────────

def process_in_batches(segments: list[dict], user_prompt: str, batch_size: int = 10) -> list[dict]:
    """Process segments in batches concurrently."""
    verified_results = []
    
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    
    def process_batch(idx_and_batch):
        idx, batch = idx_and_batch
        logger.info("Processing batch %d (out of %d)...", idx + 1, len(batches))
        verified_batch = _verify_batch(batch, user_prompt)
        return idx, batch, verified_batch

    # Use ThreadPoolExecutor to run batches in parallel
    results_by_batch = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for result in executor.map(process_batch, enumerate(batches)):
            results_by_batch.append(result)
            
    for idx, batch, verified_batch in results_by_batch:
        # Merge results, fallback to original if missing
        for original_seg in batch:
            sid = original_seg["segment_id"]
            # Find the verified version of this segment
            # The LLM is instructed to maintain segment IDs
            v_text = None
            for v_seg in verified_batch:
                if v_seg.get("segment_id") == sid:
                    v_text = v_seg.get("segment_text")
                    break
            
            if v_text is not None and v_text.strip():
                verified_results.append({
                    "segment_id": sid,
                    "segment_text": v_text.strip()
                })
            else:
                logger.warning("Segment %d missing from verification or empty, using original.", sid)
                verified_results.append({
                    "segment_id": sid,
                    "segment_text": original_seg["segment_text"]
                })
                
    # Sort by segment_id just in case
    verified_results.sort(key=lambda x: x["segment_id"])
    return verified_results

VERIFICATION_SYSTEM_PROMPT = """You are the FINAL VERIFIER in an edtech verification console. This is the last step before publishing.
Your job is to verify, format, or correct questions according strictly to the USER VERIFICATION INSTRUCTIONS.

CRITICAL: The USER VERIFICATION INSTRUCTIONS (provided in the user prompt) take absolute precedence. If the user asks you to format, translate, tag, adjust tone, or modify the questions in any specific way, you MUST follow their instructions exactly, even if it contradicts the General Guidelines below.

General Guidelines (apply unless overridden by user instructions):
• Strict in using evidence-based judgement which is in line with the latest update of that concept or Act.
• Exam-oriented (tight framing, no weird/AI phrasing)
• Zero tolerance for conceptual errors
• No guesswork: if a question cannot be reliably corrected, reject it silently and give back the exact given text without any changes

Rules:
- You will receive a list of segments, each prefixed with SEGMENT_ID: <id>.
- Only modify the segments based on the user's prompt and guidelines.
- Do not invent new facts unless the user prompt specifically asks you to.
- Maintain the original segment IDs.
- You MUST output your results as a JSON array of objects.
- Each object must have "segment_id" (integer) and "segment_text" (string - the modified text).

Your output MUST be a JSON object containing a "verified_segments" array.
"""

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _call_openai_json_verifier(system_prompt: str, user_prompt: str) -> str:
    client = _get_client()
    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        temperature=0.0,
        top_p=1.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()

def _verify_batch(batch: list[dict], user_instructions: str) -> list[dict]:
    # Construct batch string
    batch_text = ""
    for seg in batch:
        batch_text += f"SEGMENT_ID: {seg['segment_id']}\n{seg['segment_text']}\n\n"
        
    full_prompt = ""
    if user_instructions and user_instructions.strip():
        full_prompt += f"USER VERIFICATION INSTRUCTIONS:\n{user_instructions}\n\n"
        
    full_prompt += f"SEGMENTS TO VERIFY:\n{batch_text}"
    
    system_prompt = VERIFICATION_SYSTEM_PROMPT + "\n\nOutput a JSON object with a 'verified_segments' key containing the array."
    
    try:
        raw_json = _call_openai_json_verifier(system_prompt, full_prompt)
        data = json.loads(raw_json)
        
        if isinstance(data, dict):
            if "verified_segments" in data:
                return data["verified_segments"]
            for k, v in data.items():
                if isinstance(v, list):
                    return v
            return []
        elif isinstance(data, list):
            return data
        return []
    except Exception:
        logger.exception("Verification batch failed.")
        return []


