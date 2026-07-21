from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def test_itinerary_quality():
    # 1. The input we are testing
    input_text = "I want a Tainan food tour with NO seafood."
    
    # 2. Run your actual agent
    actual_output = "Visit Anping for Fish Balls and Shrimp Rolls..." # OOPS!
    
    # 3. Define the metric (Is it relevant? Did it follow constraints?)
    # In 2026, we use LLM-as-a-Judge to score this 0-1
    metric = AnswerRelevancyMetric(threshold=0.7)
    test_case = LLMTestCase(
        input=input_text,
        actual_output=actual_output,
        retrieval_context=["User is allergic to seafood. Tainan has beef soup and noodles as alternatives."]
    )

    evaluate([test_case], [metric])

if __name__ == "__main__":
    test_itinerary_quality()