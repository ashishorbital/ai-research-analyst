import json
import os

from dotenv import load_dotenv
from groq import Groq

from .models import Evidence, Source


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


# ============================================================
# JSON PARSER
# ============================================================

def parse_json_response(text: str) -> dict:
    """
    Safely parse JSON returned by the LLM.

    Handles:
    - normal JSON
    - ```json ... ``` code fences
    - extra text surrounding the JSON object
    """

    if not text:
        raise ValueError("Empty model response.")

    text = text.strip()

    # --------------------------------------------------------
    # Remove markdown code fences
    # --------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # --------------------------------------------------------
    # First attempt: parse the entire response
    # --------------------------------------------------------

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------------------
    # Second attempt: find JSON object inside response
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:

        candidate = text[start:end + 1]

        try:
            return json.loads(candidate)

        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not parse valid JSON from model response."
    )


# ============================================================
# EVIDENCE NORMALIZATION
# ============================================================

def normalize_evidence(
    item: dict,
    question: str,
    source: Source
) -> Evidence:
    """
    Normalize and validate one evidence item.
    """

    # --------------------------------------------------------
    # Claim
    # --------------------------------------------------------

    claim = str(
        item.get("claim", "")
    ).strip()

    # --------------------------------------------------------
    # Evidence text
    # --------------------------------------------------------

    evidence_text = str(
        item.get("evidence", "")
    ).strip()

    # --------------------------------------------------------
    # Evidence type
    # --------------------------------------------------------

    evidence_type = str(
        item.get(
            "evidence_type",
            "inference"
        )
    ).strip().lower()

    if evidence_type not in {
        "fact",
        "inference",
        "opinion"
    }:
        evidence_type = "inference"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    try:

        confidence = float(
            item.get(
                "confidence",
                0.5
            )
        )

    except (TypeError, ValueError):

        confidence = 0.5

    # Keep confidence between 0 and 1

    confidence = max(
        0.0,
        min(1.0, confidence)
    )

    # --------------------------------------------------------
    # Prevent overconfidence
    # --------------------------------------------------------

    if (
        confidence >= 0.95
        and evidence_type != "fact"
    ):
        confidence = 0.85

    # --------------------------------------------------------
    # Create Pydantic object
    # --------------------------------------------------------

    return Evidence(
        research_question=question,
        claim=claim,
        evidence=evidence_text,
        source_url=source.url,
        source_title=source.title,
        confidence=confidence,
        evidence_type=evidence_type,
    )


# ============================================================
# MODEL CALL
# ============================================================

def call_model(prompt: str) -> str:
    """
    Send a prompt to Groq and return the model response.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.0,
    )

    return response.choices[0].message.content or ""


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

def extract_evidence(
    question: str,
    source: Source
) -> list[Evidence]:

    # ========================================================
    # PRIMARY PROMPT
    # ========================================================

    prompt = f"""
You are a careful research evidence analyst.

Your job is to extract evidence from ONE source that helps answer
the research question.

RESEARCH QUESTION:

{question}


SOURCE TITLE:

{source.title}


SOURCE URL:

{source.url}


SOURCE CONTENT:

{source.content}


IMPORTANT RULES:

1. Use ONLY information explicitly supported by the source.

2. Do NOT use outside knowledge.

3. Do NOT invent facts, statistics, dates, events, or conclusions.

4. Keep each evidence item focused on ONE claim.

5. Separate facts, inferences, and opinions.


CLASSIFICATION RULES:


FACT:

The source explicitly states or directly reports the claim.


INFERENCE:

The source provides evidence from which the claim can reasonably
be inferred, but the source does not directly state the conclusion.


OPINION:

The claim is someone's judgment, prediction, expectation,
interpretation, or recommendation.


IMPORTANT:

Predictions and conditional statements are NOT automatically facts.

For example:

"The shortage could reduce production."

This should generally be classified as an inference.

Statements containing words such as:

- could
- may
- might
- likely
- expected
- potential
- risk
- threaten
- would
- if

should be carefully evaluated and should usually NOT be classified
as established facts unless the source explicitly presents them
as established facts.


CONFIDENCE SCORING:

0.90 - 1.00
Directly stated and clearly supported by the source.

0.75 - 0.89
Strongly supported but requires some interpretation.

0.50 - 0.74
Reasonable inference or indirect evidence.

Below 0.50
Weak or uncertain evidence.


Do NOT automatically assign 0.95.


If the source attributes a statement to another person or
organization, preserve that attribution.

For example:

"Analysts expect Tesla's margins to decline."

Do NOT rewrite this as:

"Tesla's margins will decline."


Instead preserve the attribution:

"Analysts expect Tesla's margins to decline."


Ignore irrelevant information.

Extract at most 8 strong evidence items.

If the source contains no useful evidence, return an empty list.


RETURN ONLY VALID JSON.

Use exactly this structure:


{{
    "evidence": [
        {{
            "claim": "short claim",
            "evidence": "what the source actually says",
            "confidence": 0.80,
            "evidence_type": "fact"
        }}
    ]
}}


Do not include markdown.

Do not include explanations outside the JSON.
"""

    # ========================================================
    # FIRST ATTEMPT
    # ========================================================

    try:

        content = call_model(prompt)

        if not content.strip():

            raise ValueError(
                "Model returned an empty response."
            )

        data = parse_json_response(content)

    except Exception as first_error:

        print(
            "  ↻ Retrying evidence extraction..."
        )

        # ====================================================
        # RETRY PROMPT
        # ====================================================

        retry_prompt = f"""
Extract useful evidence from the source below.

RESEARCH QUESTION:

{question}


SOURCE TITLE:

{source.title}


SOURCE CONTENT:

{source.content}


Return ONLY valid JSON.

Use EXACTLY this structure:


{{
  "evidence": [
    {{
      "claim": "claim supported by source",
      "evidence": "supporting information from source",
      "confidence": 0.80,
      "evidence_type": "fact"
    }}
  ]
}}


RULES:

- Only use information from the source.
- Do not invent information.
- FACT = directly stated by source.
- INFERENCE = reasonably derived from source.
- OPINION = someone's judgment or prediction.
- Predictions are usually inference or opinion.
- Conditional statements are usually inference or opinion.
- Confidence must be between 0 and 1.
- Do not automatically use 0.95.
- Maximum 6 evidence items.
- Return an empty list if there is no useful evidence.
- No markdown.
- No explanation outside JSON.
"""

        try:

            content = call_model(
                retry_prompt
            )

            if not content.strip():

                print(
                    "  ⚠ Retry also returned "
                    "an empty response."
                )

                return []

            data = parse_json_response(
                content
            )

        except Exception as retry_error:

            print(
                f"  ⚠ Retry failed: "
                f"{retry_error}"
            )

            return []

    # ========================================================
    # EXTRACT RAW EVIDENCE
    # ========================================================

    raw_evidence = data.get(
        "evidence",
        []
    )

    # ========================================================
    # VALIDATE LIST
    # ========================================================

    if not isinstance(
        raw_evidence,
        list
    ):

        print(
            "  ⚠ Invalid evidence format "
            "returned by model."
        )

        return []

    # ========================================================
    # BUILD EVIDENCE OBJECTS
    # ========================================================

    results = []

    for item in raw_evidence:

        if not isinstance(
            item,
            dict
        ):
            continue

        try:

            evidence = normalize_evidence(
                item,
                question,
                source
            )

            # Skip incomplete evidence

            if not evidence.claim:
                continue

            if not evidence.evidence:
                continue

            results.append(
                evidence
            )

        except Exception as e:

            print(
                f"  ⚠ Skipping invalid "
                f"evidence item: {e}"
            )

    return results


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_evidence(
    evidence_items: list[Evidence]
) -> list[Evidence]:
    """
    Remove duplicate evidence items based on their claims.
    """

    seen = set()

    unique = []

    for item in evidence_items:

        key = (
            item.claim
            .lower()
            .strip()
        )

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)

        unique.append(item)

    return unique