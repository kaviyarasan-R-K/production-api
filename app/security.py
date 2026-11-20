import re, unicodedata
from typing import Optional
from langsmith import traceable

class InputSanitizer:

    INJECTION_PATTERNS = [
    # 1. Instruction Overrides (fixed spelling + flexible phrasing)
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions?|directives?|rules?|prompts?)",
    r"(?:disregard|forget|override|bypass)\s+(?:all\s+|any\s+)?(?:previous|prior|above|system)\s+(?:instructions?|rules?|context)",
    r"new\s+(?:instructions?|directives?|rules?|system\s+prompt)\s*:",

    # 2. Persona & Jailbreak Hijacking
    r"(?:you\s+are\s+now|act\s+as|pretend\s+to\s+be|switch\s+to)\s+(?:a\s+|an\s+)?(?:unrestricted|unfiltered|developer|admin|DAN)",
    r"\b(?:DAN|do\s+anything\s+now|jailbreak|developer\s+mode)\b",

    # 3. System Prompt & Secret Extraction
    r"(?:reveal|show|print|output|repeat|dump|display)\s+(?:your\s+|the\s+)?(?:system\s+prompt|initial\s+instructions?|hidden\s+rules?)",
    r"\bsystem\s*prompt\b",

    # 4. Special Delimiter / Token Spoofing
    r"<\s*\|\s*(?:im_start|im_end|system|endoftext)\s*\|\s*>",
    r"\[/?(?:INST|SYS|SYSTEM)\]",
    r"###\s*(?:System|Instruction|Human|Assistant)\s*:",
]

    def __init__(self):
        self.patterns = [
            re.compile(p, re.IGNORECASE)
            for p in self.INJECTION_PATTERNS
        ]

    def check(self, text: str) -> tuple[bool, Optional[str]]:
        for pattern in self.patterns:
            if pattern.search(text):
                return False, "Blocked: Injection Dectected"
        return True, None

    def clean(self, text: str) -> str:
        # Remove markdown-style delimiter lines (--- or ===)
        text = re.sub(r'[-]{3,}', '', text)
        text = re.sub(r'[=]{3,}', '', text)
        
        # Break up template injection braces by adding a space inside them
        text = text.replace('{{', '{ {').replace('}}', '} }')
        
        return text.strip()

class PIIDetector:
    PATTERNS = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "SSN": re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{4}[-\s]?\d{6}[-\s]?\d{5}\b"),
        "PHONE": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"),
    }

    MASK_MAP = {
        "email": "[EMAIL REDACTED]",
        "phone": "[PHONE REDACTED]",
        "ssn": "[SSN REDACTED]",
        "credit_card": "[CARD REDACTED]"

    }

    def detect(self, text: str) -> dict[str, list[str]]:
        found = {}
        for pii_type, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found[pii_type] = matches
        return found

    def mask(self, text: str) -> str:
        masked = text
        for pii_type, pattern in self.PATTERNS.items():
            replacement = self.MASK_MAP.get(pii_type.lower(), f"[{pii_type} REDACTED]")
            masked = pattern.sub(replacement, masked)
        return masked



class OutputValidator:
    HARMFUL_PATTERNS = [
        re.compile(r"here(?:'s|\s+is)\s+(?:how|the\s+way)\s+to\s+(?:hack|steal|attack)", re.I),
        re.compile(r"password\s+is\s+", re.I),
        re.compile(r"api[_\s]?key\s*[:=]", re.I),
    ]

    def __init__(self): 
        self.pii_detector = PIIDetector()

    def validate(self, output: str) -> tuple[str, list[str]]:
        warnings = []

        # 1. Detect and mask any PII leaked in the LLM response
        pii_found = self.pii_detector.detect(output)
        if pii_found:
            output = self.pii_detector.mask(output)
            warnings.append(f"PII masked in output: {list(pii_found.keys())}")  # Added () to .keys()

        # 2. Block responses containing harmful instructions or credentials
        for pattern in self.HARMFUL_PATTERNS:
            if pattern.search(output):
                output = "[Response blocked: Potentially harmful content]"
                warnings.append("Harmful content blocked")
                break

        return output, warnings

class SecurityPipeline:
    
    def __init__(self):
        self.sanitizer = InputSanitizer()
        self.pii_detector = PIIDetector()
        self.output_validator = OutputValidator()

    @traceable(name="security check point")
    def check_input(self, text: str) -> tuple[bool, str, list[str]]:
        notes =[]

        is_safe, reason =  self.sanitizer.check(text)
        if not is_safe:
            return False, "", [reason]

        cleaned = self.sanitizer.clean(text)

        pii_found = self.pii_detector.detect(cleaned)
        if pii_found:
            cleaned = self.pii_detector.mask(cleaned)
            notes.append(f"Input PII masked: {list(pii_found.keys())}")
        return True, cleaned, notes

    @traceable(name = "Security_check_output")
    def check_output(self, text: str) -> tuple[bool, str, list[str]]:

        return self.output_validator.validate(text)



