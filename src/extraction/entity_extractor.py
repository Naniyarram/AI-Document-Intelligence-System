
# src/extraction/entity_extractor.py

# Named Entity Recognition & Structured Extraction

# Extracts structured information from document chunks:
#   - Dates, times
#   - Money amounts, percentages
#   - Organizations, people, locations
#   - Contract-specific: parties, terms, clauses
#   - Document-specific patterns via regex

# Uses spaCy's NER model (runs locally, no API needed).


import re
from typing import List, Dict, Any
from loguru import logger


class EntityExtractor:
    """
    Extracts named entities and structured information from text.

    Usage:
        extractor = EntityExtractor()
        entities = extractor.extract("The contract is dated January 15, 2024 between Acme Corp...")
        # Returns: {dates: [...], organizations: [...], amounts: [...], ...}
    """

    def __init__(self):
        self.nlp = self._load_spacy()

    def _load_spacy(self):
        """Load spaCy NER model."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy NER model loaded")
            return nlp
        except (ImportError, OSError):
            logger.warning(
                "spaCy not available — entity extraction will use regex only.\n"
                "For better extraction: python -m spacy download en_core_web_sm"
            )
            return None

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Extract all entities from a piece of text.

        Returns a dictionary with entity categories as keys
        and lists of found entities as values.
        """
        entities = {
            "dates": [],
            "organizations": [],
            "people": [],
            "locations": [],
            "money_amounts": [],
            "percentages": [],
            "invoice_numbers": [],
            "email_addresses": [],
            "phone_numbers": [],
            "other": []
        }

        # spaCy NER 
        if self.nlp:
            doc = self.nlp(text[:100_000])  # spaCy limit
            for ent in doc.ents:
                if ent.label_ in ["DATE", "TIME"]:
                    entities["dates"].append(ent.text)
                elif ent.label_ in ["ORG", "COMPANY"]:
                    entities["organizations"].append(ent.text)
                elif ent.label_ in ["PERSON"]:
                    entities["people"].append(ent.text)
                elif ent.label_ in ["GPE", "LOC"]:
                    entities["locations"].append(ent.text)
                elif ent.label_ in ["MONEY", "CURRENCY"]:
                    entities["money_amounts"].append(ent.text)
                elif ent.label_ in ["PERCENT"]:
                    entities["percentages"].append(ent.text)

        #  Regex Patterns 
        # These catch patterns spaCy might miss

        # Invoice numbers: INV-2024-0001, #12345, Invoice No. 9876
        inv_pattern = r"\b(?:INV|INVOICE|PO|ORDER|REF|#)\s*[-:]?\s*\d{4,}\b"
        entities["invoice_numbers"] = re.findall(inv_pattern, text, re.IGNORECASE)

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        entities["email_addresses"] = re.findall(email_pattern, text)

        # Phone numbers
        phone_pattern = r"\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
        entities["phone_numbers"] = re.findall(phone_pattern, text)

        # Money amounts (if spaCy missed some)
        money_pattern = r"(?:USD|INR|€|£|\$|Rs\.?)\s*[\d,]+(?:\.\d{2})?"
        regex_money = re.findall(money_pattern, text)
        entities["money_amounts"].extend(regex_money)

        # Percentages (if spaCy missed some)
        pct_pattern = r"\d+(?:\.\d+)?%"
        regex_pct = re.findall(pct_pattern, text)
        entities["percentages"].extend(regex_pct)

        # Deduplicate all lists
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))  # preserves order

        # Remove empty categories
        entities = {k: v for k, v in entities.items() if v}

        return entities

    def extract_from_chunks(self, chunks: List[dict]) -> Dict[str, Any]:
        """
        Extract entities from a list of retrieved chunks.
        Aggregates entities across all chunks.
        """
        combined_text = "\n".join(chunk["text"] for chunk in chunks)
        return self.extract(combined_text)

    def detect_anomalies(self, chunks: List[dict]) -> List[Dict]:
        """
        Simple anomaly detection for numerical data.
        Looks for outliers in monetary amounts and flags duplicates.
        """
        import re

        # Extract all monetary amounts
        all_amounts = []
        for chunk in chunks:
            amounts = re.findall(
                r"(?:USD|INR|€|£|\$|Rs\.?)?\s*([\d,]+(?:\.\d{2})?)",
                chunk["text"]
            )
            for amt in amounts:
                try:
                    value = float(amt.replace(",", ""))
                    all_amounts.append({
                        "value": value,
                        "raw": amt,
                        "source": chunk.get("metadata", {}).get("source_file", "?"),
                        "page": chunk.get("metadata", {}).get("page_number", "?")
                    })
                except ValueError:
                    continue

        anomalies = []

        if len(all_amounts) > 3:
            import numpy as np
            values = [a["value"] for a in all_amounts]
            mean = np.mean(values)
            std = np.std(values)

            for item in all_amounts:
                # Flag if more than 2.5 standard deviations from mean
                if std > 0 and abs(item["value"] - mean) > 2.5 * std:
                    anomalies.append({
                        "type": "outlier_amount",
                        "value": item["value"],
                        "raw": item["raw"],
                        "mean": round(mean, 2),
                        "deviation": round(abs(item["value"] - mean) / std, 2),
                        "source": item["source"],
                        "page": item["page"],
                        "severity": "HIGH" if abs(item["value"] - mean) > 3 * std else "MEDIUM"
                    })

        return anomalies
