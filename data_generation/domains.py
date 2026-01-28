import random
from typing import Tuple

COMPLEXITY_LEVELS = {
    "simple": {"min_nodes": 3, "max_nodes": 4, "description": "simple"},
    "moderate": {"min_nodes": 5, "max_nodes": 6, "description": "moderate"},
    "complex": {"min_nodes": 7, "max_nodes": 8, "description": "complex"},
}

DOMAINS = {
    "business_process": [
        "customer onboarding",
        "invoice processing",
        "budget approval",
        "vendor evaluation",
        "contract negotiation",
    ],
    "sales": [
        "lead qualification",
        "deal closing",
        "proposal creation",
        "customer follow-up",
        "pipeline management",
    ],
    "marketing": [
        "campaign launch",
        "content approval",
        "social media posting",
        "email marketing",
        "brand review",
    ],
    "hr": [
        "hiring process",
        "performance review",
        "leave approval",
        "employee offboarding",
        "training enrollment",
    ],
    "finance": [
        "expense reimbursement",
        "financial reporting",
        "audit preparation",
        "tax filing",
        "budget planning",
    ],
    "software_development": [
        "code review",
        "bug fixing",
        "feature deployment",
        "sprint planning",
        "release management",
    ],
    "devops": [
        "CI/CD pipeline",
        "server deployment",
        "monitoring setup",
        "incident response",
        "backup restoration",
    ],
    "data_engineering": [
        "ETL pipeline",
        "data validation",
        "schema migration",
        "data backup",
        "query optimization",
    ],
    "security": [
        "access request",
        "vulnerability assessment",
        "incident handling",
        "security audit",
        "password reset",
    ],
    "it_support": [
        "ticket resolution",
        "hardware provisioning",
        "software installation",
        "network troubleshooting",
        "account creation",
    ],
    "patient_care": [
        "patient admission",
        "treatment planning",
        "medication dispensing",
        "discharge process",
        "follow-up scheduling",
    ],
    "medical_admin": [
        "appointment booking",
        "insurance verification",
        "medical records transfer",
        "billing process",
        "referral handling",
    ],
    "lab_process": [
        "sample collection",
        "test processing",
        "result reporting",
        "quality control",
        "equipment calibration",
    ],
    "student_services": [
        "course registration",
        "grade submission",
        "scholarship application",
        "graduation clearance",
        "transcript request",
    ],
    "academic": [
        "research proposal",
        "thesis submission",
        "peer review",
        "publication process",
        "grant application",
    ],
    "school_admin": [
        "enrollment verification",
        "attendance tracking",
        "parent notification",
        "report card generation",
        "fee collection",
    ],
    "manufacturing": [
        "production planning",
        "quality inspection",
        "inventory restocking",
        "equipment maintenance",
        "defect handling",
    ],
    "supply_chain": [
        "order fulfillment",
        "shipment tracking",
        "supplier selection",
        "demand forecasting",
        "returns processing",
    ],
    "warehouse": [
        "receiving goods",
        "inventory counting",
        "order picking",
        "shipping preparation",
        "storage allocation",
    ],
    "legal": [
        "contract review",
        "legal research",
        "case filing",
        "document notarization",
        "dispute resolution",
    ],
    "compliance": [
        "policy approval",
        "audit response",
        "risk assessment",
        "regulatory filing",
        "violation handling",
    ],
    "customer_support": [
        "complaint handling",
        "refund processing",
        "escalation management",
        "feedback collection",
        "service recovery",
    ],
    "technical_support": [
        "issue diagnosis",
        "remote troubleshooting",
        "patch deployment",
        "system restoration",
        "user training",
    ],
    "real_estate": [
        "property listing",
        "buyer qualification",
        "offer negotiation",
        "closing process",
        "inspection coordination",
    ],
    "property_management": [
        "tenant screening",
        "lease renewal",
        "maintenance request",
        "rent collection",
        "move-out inspection",
    ],
    "restaurant": [
        "order taking",
        "food preparation",
        "table management",
        "payment processing",
        "inventory ordering",
    ],
    "hotel": [
        "reservation booking",
        "guest check-in",
        "room service",
        "housekeeping assignment",
        "check-out process",
    ],
    "government": [
        "permit application",
        "license renewal",
        "public records request",
        "benefit eligibility",
        "civic complaint",
    ],
    "emergency_services": [
        "911 dispatch",
        "incident reporting",
        "resource allocation",
        "evacuation procedure",
        "recovery coordination",
    ],
    "personal_finance": [
        "loan application",
        "investment decision",
        "insurance claim",
        "retirement planning",
        "debt payment",
    ],
    "event_planning": [
        "venue selection",
        "vendor booking",
        "guest management",
        "event execution",
        "post-event review",
    ],
    "travel": [
        "trip planning",
        "booking confirmation",
        "itinerary creation",
        "travel reimbursement",
        "visa application",
    ],
}


def get_random_domain_topic() -> Tuple[str, str, dict]:
    domain = random.choice(list(DOMAINS.keys()))
    topic = random.choice(DOMAINS[domain])
    complexity = random.choice(list(COMPLEXITY_LEVELS.values()))
    return domain, topic, complexity


def get_all_domain_topics() -> list[Tuple[str, str]]:
    pairs = []
    for domain, topics in DOMAINS.items():
        for topic in topics:
            pairs.append((domain, topic))
    return pairs
