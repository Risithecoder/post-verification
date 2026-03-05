"""
diff_service.py
───────────────
Generates side-by-side HTML diff comparisons between original
and verified document text.
"""

from __future__ import annotations

import difflib
import html
import logging

logger = logging.getLogger(__name__)


def generate_diff_html(original: str, verified: str) -> dict:
    """
    Compare original and verified text line-by-line and produce
    HTML-highlighted diffs for the comparison page.

    Returns a dict with:
      - original_html: original text with deletions highlighted
      - verified_html: verified text with additions highlighted
      - combined_html: unified diff view
      - stats: dict with counts of additions, deletions, changes
    """
    orig_lines = original.splitlines()
    veri_lines = verified.splitlines()

    # Compute unified diff
    differ = difflib.ndiff(orig_lines, veri_lines)
    diff_lines = list(differ)

    # Build highlighted HTML for original (left) side
    original_html_parts: list[str] = []
    verified_html_parts: list[str] = []
    combined_html_parts: list[str] = []

    additions = 0
    deletions = 0
    changes = 0

    for line in diff_lines:
        code = line[:2]
        text = html.escape(line[2:])

        if code == "  ":
            # Unchanged line
            original_html_parts.append(f'<div class="diff-line">{text}</div>')
            verified_html_parts.append(f'<div class="diff-line">{text}</div>')
            combined_html_parts.append(f'<div class="diff-line">{text}</div>')
        elif code == "- ":
            # Deleted from original
            deletions += 1
            original_html_parts.append(
                f'<div class="diff-line diff-deleted">{text}</div>'
            )
            combined_html_parts.append(
                f'<div class="diff-line diff-deleted">- {text}</div>'
            )
        elif code == "+ ":
            # Added in verified
            additions += 1
            verified_html_parts.append(
                f'<div class="diff-line diff-added">{text}</div>'
            )
            combined_html_parts.append(
                f'<div class="diff-line diff-added">+ {text}</div>'
            )
        elif code == "? ":
            # Hint line (character-level change indicator)
            changes += 1
            combined_html_parts.append(
                f'<div class="diff-line diff-hint">{text}</div>'
            )

    stats = {
        "additions": additions,
        "deletions": deletions,
        "changes": changes,
        "total_original": len(orig_lines),
        "total_verified": len(veri_lines),
    }

    logger.info(
        "Diff stats: %d additions, %d deletions, %d changes",
        additions, deletions, changes,
    )

    return {
        "original_html": "\n".join(original_html_parts),
        "verified_html": "\n".join(verified_html_parts),
        "combined_html": "\n".join(combined_html_parts),
        "stats": stats,
    }
