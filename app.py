import gradio as gr
from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

# Download the tokenizer (required for the first time)
nltk.download('punkt_tab')

# Define the summarizer function
def summarize_text(input_text, num_points):
    # Initialize the summarization pipeline
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

    # Break the input text into chunks to fit model limits
    max_chunk_size = 512  # Adjust based on model's max token limit
    chunks = [input_text[i:i + max_chunk_size] for i in range(0, len(input_text), max_chunk_size)]

    # Summarize each chunk and store results
    summarized_chunks = [summarizer(chunk, max_length=50, min_length=10, do_sample=False)[0]['summary_text'] for chunk in chunks]

    # Combine all summaries
    combined_summary = " ".join(summarized_chunks)

    # Use NLTK to tokenize sentences, ensuring clear segmentation
    sentences = sent_tokenize(combined_summary)

    # Select the required number of unique points
    unique_points = []
    for sentence in sentences:
        if len(unique_points) < num_points and sentence not in unique_points:
            unique_points.append(sentence.strip().capitalize())

    # Ensure the number of points matches the user's request
    if len(unique_points) < num_points:
        extra_sentences = [s.strip().capitalize() for s in sentences if s not in unique_points]
        unique_points.extend(extra_sentences[:num_points - len(unique_points)])

    # Return the summary as bullet points
    return "\n".join(f"- {point}" for point in unique_points)

# Define Gradio interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("### Text Summarizer")
        gr.Markdown("Input your text and specify the number of points you want to summarize.")

        # Input text
        input_text = gr.TextArea(label="Enter Text to Summarize", placeholder="Paste your text here...", lines=10)

        # Input box for number of points
        num_points = gr.Number(label="Number of Summary Points", value=3)

        # Output text
        output_text = gr.TextArea(label="Summarized Text", placeholder="Your summarized text will appear here...", lines=10)

        # Summarize button
        summarize_button = gr.Button("Summarize")

        # Define function for button
        summarize_button.click(fn=summarize_text, inputs=[input_text, num_points], outputs=output_text)

    return demo

# Run the app
if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()