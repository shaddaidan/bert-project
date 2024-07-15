from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

def perform_ner(text):
    # Load pre-trained model and tokenizer
    model_name = "dbmdz/bert-large-cased-finetuned-conll03-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    # Create NER pipeline
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Perform NER
    results = ner_pipeline(text)

    # Process and print results
    for entity in results:
        print(f"Entity: {entity['word']}")
        print(f"Type: {entity['entity_group']}")
        print(f"Confidence: {entity['score']:.4f}")
        print("---")

# Example usage
text = """
Apple Inc. is planning to open a new office in New York City next year. 
The company's CEO, Tim Cook, announced this during his visit to Microsoft's headquarters in Redmond, Washington.
"""

perform_ner(text)

#  this is not yet working so we would have to come and work on this later