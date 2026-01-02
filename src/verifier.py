"""
TCM-Sage Self-Critique Verifier

This module implements the reflective verification step (inspired by Self-RAG).
It allows the system to audit its own generated answers against the retrieved
context to ensure factual accuracy and prevent hallucinations.
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class VerificationResult(BaseModel):
    """Result of the self-critique verification process."""
    is_faithful: bool = Field(description="Whether the answer is strictly based on the provided context")
    is_complete: bool = Field(description="Whether the answer fully addresses the user's question")
    critique: str = Field(description="Detailed explanation of the faithfulness and completeness check")
    confidence_score: float = Field(description="0.0 to 1.0 score of how trustworthy the answer is")

class SelfCritiqueVerifier:
    """
    Reflective verifier that audits generated answers.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.parser = JsonOutputParser(pydantic_object=VerificationResult)
        
        self.prompt = ChatPromptTemplate.from_template(
            """You are a specialized TCM Auditor. Your task is to verify an AI-generated answer 
against the provided classical TCM context.

### Question
{question}

### Retrieved Context (Source Documents & Knowledge Graph Facts)
{context}

### Generated Answer
{answer}

### Instructions
1. Check for FAITHFULNESS: Is every medicinal claim, formula name, or symptom association mentioned in the answer directly supported by the context? Flag any hallucinations.
2. Check for COMPLETENESS: Does the answer address use all relevant information from the context to answer the specific question?
3. Assign a trust score (0.0 to 1.0).

Return your critique in JSON format with the following keys:
- is_faithful (boolean)
- is_complete (boolean)
- critique (string)
- confidence_score (float)

Your response must be ONLY the JSON object.
"""
        )
        self.chain = self.prompt | self.llm | self.parser

    def verify(self, question: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Audit the answer against the context.
        """
        return self.chain.invoke({
            "question": question,
            "context": context,
            "answer": answer
        })
