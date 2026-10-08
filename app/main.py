from .planner import create_research_plan
from .researcher import search_web, fetch_page
from .evidence import extract_evidence
from .models import Source, Evidence
from .evidence import (
    extract_evidence,
    deduplicate_evidence
)


def main():

    query = input("Research question: ")

    # ==================================================
    # 1. CREATE RESEARCH PLAN
    # ==================================================

    print("\nCreating research plan...")

    plan = create_research_plan(query)

    print("\nRESEARCH PLAN")
    print("=" * 70)

    for i, question in enumerate(plan.questions, 1):

        print(f"\n{i}. {question.question}")
        print(f"   Purpose: {question.purpose}")

    # ==================================================
    # 2. RESEARCH
    # ==================================================

    all_evidence: list[Evidence] = []

    print("\n\nSTARTING RESEARCH")
    print("=" * 70)

    for i, research_question in enumerate(
        plan.questions,
        1
    ):

        print(
            f"\n\n[{i}/{len(plan.questions)}] "
            f"{research_question.question}"
        )

        # ------------------------------------------------
        # SEARCH
        # ------------------------------------------------

        sources = search_web(
            research_question.question,
            max_results=3
        )

        if not sources:

            print("  ⚠ No sources found.")

            continue

        print(
            f"  Found {len(sources)} sources."
        )

        # ------------------------------------------------
        # READ + EXTRACT
        # ------------------------------------------------

        for j, source in enumerate(
            sources,
            1
        ):

            print(
                f"\n  Source {j}: {source.title}"
            )

            print(
                f"  URL: {source.url}"
            )

            # Fetch webpage
            content = fetch_page(
                source.url
            )

            if not content:

                print(
                    "  ⚠ Could not read source."
                )

                continue

            print(
                f"  ✓ Read {len(content):,} characters."
            )

            # Create complete Source object
            source = Source(
                title=source.title,
                url=source.url,
                content=content
            )

            # Extract evidence
            try:

                evidence = extract_evidence(
                    research_question.question,
                    source
                )

            except Exception as e:

                print(
                    f"  ⚠ Evidence extraction failed: {e}"
                )

                continue

            if not evidence:

                print(
                    "  No useful evidence found."
                )

                continue

            print(
                f"  ✓ Extracted {len(evidence)} "
                f"evidence item(s)."
            )

            # Add to global ledger
            all_evidence.extend(
                evidence
            )

            all_evidence = deduplicate_evidence(
                all_evidence
            )

    # ==================================================
    # 3. DISPLAY EVIDENCE LEDGER
    # ==================================================

    print("\n\n")
    print("=" * 70)
    print("EVIDENCE LEDGER")
    print("=" * 70)

    if not all_evidence:

        print("\nNo evidence was collected.")

        return

    for i, item in enumerate(
        all_evidence,
        1
    ):

        print(f"\n[{i}] {item.claim}")

        print(
            f"    Evidence: {item.evidence}"
        )

        print(
            f"    Type: {item.evidence_type}"
        )

        print(
            f"    Confidence: {item.confidence:.2f}"
        )

        print(
            f"    Source: {item.source_title}"
        )

        print(
            f"    URL: {item.source_url}"
        )

    print("\n")
    print(
        f"Total evidence items: "
        f"{len(all_evidence)}"
    )


if __name__ == "__main__":
    main()