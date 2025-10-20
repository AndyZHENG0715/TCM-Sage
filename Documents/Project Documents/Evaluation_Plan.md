# TCM-Sage: Evaluation Plan & Pilot Testing Strategy

This document outlines the methodology for evaluating the effectiveness of the TCM-Sage system and the strategy for conducting pilot tests with target users.

## 1. Evaluation Objectives

The primary goal of the evaluation is to quantitatively and qualitatively measure the system's performance against its core objectives: providing accurate, fast, and trustworthy evidence-based answers for TCM practitioners.

## 2. Evaluation Metrics

To ensure a comprehensive assessment, the evaluation will be based on the following key metrics, as suggested during project supervision.

### 2.1. Quantitative Metrics

These metrics will be measured internally using a predefined set of test questions.

* **Response Latency:**

    - **Definition:** The time elapsed from submitting a query to receiving the complete generated response.

    - **Measurement:** The system will log the start and end time for each query. The average latency will be calculated in seconds.

    - **Success Criterion:** Average latency should be under 5 seconds for a positive user experience.

* **Citation Precision:**

    - **Definition:** The percentage of retrieved and cited sources that are relevant to the user's query.

    - **Measurement:** For each test query, the retrieved source chunks will be manually checked. Precision = (Number of Relevant Chunks Retrieved) / (Total Number of Chunks Retrieved).

    - **Success Criterion:** Precision score should be above 90%.

* **Answer Faithfulness:**

    - **Definition:** The degree to which the generated answer is factually consistent with the information present in the cited source text.

    - **Measurement:** Generated answers for test questions will be manually compared against their cited sources and rated on a 1-5 scale (1: Contradictory, 3: Related but not fully supported, 5: Fully supported).

    - **Success Criterion:** The average faithfulness score should be 4.5 or higher.

### 2.2. Qualitative Metrics (via Pilot Testing)

These metrics will be gathered through user feedback during the pilot testing phase.

* **User Trust & Confidence Score:**

    - **Definition:** The user's confidence in the accuracy and reliability of the generated answer and its citation.

    - **Measurement:** A post-test survey will ask users to rate their trust in the system on a Likert scale (1: Very Low, 5: Very High).

* **System Usability & Citation Clarity:**

    - **Definition:** The ease of use of the interface and the clarity and usefulness of the provided citations.

    - **Measurement:** The survey will include questions about the user interface and open-ended questions for feedback (e.g., "Was the citation easy to understand and verify?").

## 3. Pilot Testing Plan

To gather real-world feedback, a pilot test will be conducted with members of the target user group.

* **Target Participants:**

    - **Group:** 3-5 volunteers from the Hong Kong Baptist University School of Chinese Medicine.

    - **Ideal Profile:** Senior undergraduate students (Year 3+), postgraduate students, or junior practitioners who have a foundational knowledge of classical TCM texts.

* **Testing Procedure:**

    1. **Introduction (5 mins):** A brief introduction to the TCM-Sage project, emphasizing its role as an evidence-synthesis assistant, not a diagnostic tool.

    2. **Task-Based Session (15 mins):** Participants will be given a list of 5-10 predefined clinical questions to ask the system. These questions will be designed to test different aspects of the system (e.g., single herb queries, formula comparisons, symptom analysis).

    3. **Free Exploration (10 mins):** Participants will be encouraged to ask their own questions based on their clinical or academic interests.

    4. **Feedback & Survey (10 mins):** Participants will complete a short questionnaire to provide quantitative scores for trust and usability, as well as qualitative, open-ended feedback for improvement.

* **Feedback Analysis:**

    - **Quantitative Scores:** The quantitative scores will be averaged to measure overall user satisfaction and trust.

    - **Qualitative Feedback:** The qualitative feedback will be analyzed to identify common themes, pain points, and suggestions for feature enhancements, which will guide the final refinement of the system before project submission.
