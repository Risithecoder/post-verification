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
import time
from typing import Sequence
import concurrent.futures

from openai import OpenAI, RateLimitError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
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

def process_document(text: str, user_verification_prompt: str) -> dict:
    """
    Accept the full document text, segment it, verify it in batches,
    and return a dict with the fully verified document text and debug logs.
    """
    if not text.strip():
        return {"text": "", "logs": {}}

    total_start = time.time()

    logger.info("Starting segmentation...")
    seg_start = time.time()
    raw_segments = segment_document(text)
    logger.info("Extracted %d segments in %.1fs.", len(raw_segments), time.time() - seg_start)

    segments = clean_segments(raw_segments)
    logger.info("Cleaned down to %d segments.", len(segments))

    if not segments:
        return {"text": text, "logs": {}}

    verify_start = time.time()
    verified_segments, logs = process_in_batches(segments, user_verification_prompt)
    logger.info("Batch verification completed in %.1fs.", time.time() - verify_start)

    # Reassemble
    verified_text = "\n\n".join(seg.get("segment_text", "").strip() for seg in verified_segments)

    logger.info("Total processing time: %.1fs.", time.time() - total_start)
    return {"text": verified_text, "logs": logs}


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
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
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

def process_in_batches(segments: list[dict], user_prompt: str, batch_size: int = 10) -> tuple[list[dict], dict]:
    """Process segments in batches concurrently and return (verified_results, logs)."""
    verified_results = []
    
    batches = [segments[i:i + batch_size] for i in range(0, len(segments), batch_size)]
    
    total_batches = len(batches)
    captured_logs = {}

    def process_batch(idx_and_batch):
        idx, batch = idx_and_batch
        logger.info("Processing batch %d (out of %d)...", idx + 1, total_batches)
        batch_start = time.time()
        
        # Now _verify_batch returns (verified_batch_list, batch_log_dict)
        verified_batch, batch_log = _verify_batch(batch, user_prompt, batch_idx=idx, total_batches=total_batches)
        logger.info("Batch %d completed in %.1fs.", idx + 1, time.time() - batch_start)
        return idx, batch, verified_batch, batch_log

    # Use ThreadPoolExecutor to run batches in parallel
    results_by_batch = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for result in executor.map(process_batch, enumerate(batches)):
            results_by_batch.append(result)
            
    for idx, batch, verified_batch, batch_log in results_by_batch:
        # Collect logs for the frontend
        if batch_log:
            if idx == 0:
                captured_logs["first_batch"] = batch_log
            if idx == total_batches - 1:
                captured_logs["last_batch"] = batch_log

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
    return verified_results, captured_logs

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
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
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

def _verify_batch(batch: list[dict], user_instructions: str, batch_idx: int = 0, total_batches: int = 1) -> tuple[list[dict], dict | None]:
    # Construct batch string
    batch_text = ""
    for seg in batch:
        batch_text += f"SEGMENT_ID: {seg['segment_id']}\n{seg['segment_text']}\n\n"
        
    full_prompt = ""
    if user_instructions and user_instructions.strip():
        full_prompt += f"USER VERIFICATION INSTRUCTIONS:\n{user_instructions}\n\n"
        
    full_prompt += f"SEGMENTS TO VERIFY:\n{batch_text}"
    
    system_prompt = VERIFICATION_SYSTEM_PROMPT + "\n\nOutput a JSON object with a 'verified_segments' key containing the array."
    
    batch_log = None
    
    # ── DEBUG: Log full input only for FIRST and LAST batch ───────────────────
    is_first = (batch_idx == 0)
    is_last = (batch_idx == total_batches - 1)
    should_log = is_first or is_last
    
    if should_log:
        label = "FIRST" if is_first else "LAST"
        logger.info("=" * 80)
        logger.info("[%s BATCH] INPUT — System Prompt:", label)
        logger.info(system_prompt)
        logger.info("-" * 80)
        logger.info("[%s BATCH] INPUT — User Prompt:", label)
        logger.info(full_prompt)
        logger.info("=" * 80)
        batch_log = {
            "type": label,
            "system_prompt": system_prompt,
            "user_prompt": full_prompt,
            "raw_output": ""
        }
    
    try:
        raw_json = _call_openai_json_verifier(system_prompt, full_prompt)
        
        # ── DEBUG: Log raw output only for FIRST and LAST batch ───────────────
        if should_log:
            logger.info("=" * 80)
            logger.info("[%s BATCH] OUTPUT — Raw JSON from LLM:", label)
            logger.info(raw_json)
            logger.info("=" * 80)
            if batch_log:
                batch_log["raw_output"] = raw_json
        
        data = json.loads(raw_json)
        
        if isinstance(data, dict):
            if "verified_segments" in data:
                return data["verified_segments"], batch_log
            for k, v in data.items():
                if isinstance(v, list):
                    return v, batch_log
            return [], batch_log
        elif isinstance(data, list):
            return data, batch_log
        return [], batch_log
    except Exception:
        logger.exception("Verification batch failed.")
        return [], batch_log


