# ./examples/08_email_tools.py
"""
Email Tools Example
===================
Demonstrates email validation, extraction, and domain health checks.

Run: python examples/08_email_tools.py
"""

from ollamatoolkit.tools.email import EmailTools


def main():
    tools = EmailTools()

    if not tools.available:
        print("emailtoolkit not installed. Run: pip install emailtoolkit[dns]")
        return

    print("=" * 60)
    print("EMAIL TOOLS DEMO")
    print("=" * 60)

    # =========================================================================
    # Email Validation
    # =========================================================================
    print("\n--- Email Validation ---")

    test_emails = [
        "valid@example.com",
        "Test.User+sales@Gmail.com",
        "invalid@@bad.email",
        "missing-at.com",
    ]

    for email in test_emails:
        valid = tools.is_valid(email)
        print(f"  {email}: {'✓ valid' if valid else '✗ invalid'}")

    # =========================================================================
    # Email Parsing
    # =========================================================================
    print("\n--- Email Parsing ---")

    parsed = tools.parse("Test.User+sales@Gmail.com")
    if parsed:
        print(f"  Original:   {parsed.original}")
        print(f"  Normalized: {parsed.normalized}")
        print(f"  Canonical:  {parsed.canonical}")
        print(f"  Local:      {parsed.local}")
        print(f"  Domain:     {parsed.domain}")

    # =========================================================================
    # Email Comparison
    # =========================================================================
    print("\n--- Email Comparison (Canonical Identity) ---")

    pairs = [
        ("t.e.s.t+sales@googlemail.com", "test@gmail.com"),
        ("Alice@Example.COM", "alice@example.com"),
        ("user1@gmail.com", "user2@gmail.com"),
    ]

    for email1, email2 in pairs:
        same = tools.compare(email1, email2)
        status = "SAME identity" if same else "DIFFERENT"
        print(f"  {email1} vs {email2}: {status}")

    # =========================================================================
    # Email Extraction
    # =========================================================================
    print("\n--- Email Extraction ---")

    sample_text = """
    Contact us at support@example.com or sales@company.org.
    For urgent matters, email urgent@example.com.
    Invalid emails like bad@@email or noatsign will be ignored.
    """

    emails = tools.extract(sample_text)
    print(f"  Found {len(emails)} emails:")
    for e in emails:
        print(f"    - {e.normalized} (domain: {e.domain})")

    # =========================================================================
    # Domain Health Check
    # =========================================================================
    print("\n--- Domain Health Check ---")

    domains = ["gmail.com", "example.com"]
    for domain in domains:
        health = tools.check_domain(domain)
        if health:
            print(f"  {domain}:")
            print(f"    MX records: {'Yes' if health.has_mx else 'No'}")
            print(f"    A records:  {'Yes' if health.has_a else 'No'}")
            print(f"    Deliverable: {'Yes' if health.deliverable else 'No'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n--- Text Summary ---")

    summary = tools.summary(sample_text)
    print(f"  Total found:     {summary['total_found']}")
    print(f"  Valid count:     {summary['valid_count']}")
    print(f"  Unique domains:  {summary['unique_domains']}")
    print(f"  Domains:         {summary['domains']}")


if __name__ == "__main__":
    main()
