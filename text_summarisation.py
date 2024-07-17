from transformers import pipeline

def summarize_text(text, max_length=150, min_length=50):
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Generate the summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    
    # Return the summarized text
    return summary[0]['summary_text']

# Example usage
long_text = """
The United Nations (UN) is an international organization founded in 1945 after World War II with the aim of maintaining international peace and security, developing friendly relations among nations, achieving international cooperation, and being a center for harmonizing the actions of nations. The organization is financed by assessed and voluntary contributions from its member states. The UN's mission is guided by the purposes and principles contained in its founding Charter, which includes conflict prevention, humanitarian assistance, and the promotion of human rights and sustainable development.

The UN's headquarters is in New York City, and it has other main offices in Geneva, Nairobi, Vienna, and The Hague. The organization is structured into six principal organs: the General Assembly, the Security Council, the Economic and Social Council, the Trusteeship Council, the International Court of Justice, and the UN Secretariat. The UN system includes a multitude of specialized agencies, funds, and programmes such as the World Bank Group, the World Health Organization, the World Food Programme, UNESCO, and UNICEF.

Since its founding, the UN has played a crucial role in major international issues, including decolonization, the prevention of nuclear proliferation, the promotion of human rights, and the provision of humanitarian aid in the wake of natural disasters and armed conflicts. While the organization has been successful in many of its endeavors, it has also faced criticism for its bureaucratic nature, the structure of the Security Council, and its response to certain international crises.
"""

summary = summarize_text(long_text)
print("Summary:")
print(summary)