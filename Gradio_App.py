# pip install gradio

import gradio as gr

def analyze_text(text: str, mode: str):
    """Simple processing function simulating AI text analysis."""
    if not text.strip():
        return "Please enter valid text."
    
    word_count = len(text.split())
    char_count = len(text)
    
    if mode == "Summary":
        return f"Summary: {text[:60]}... (Total: {word_count} words)"
    elif mode == "Word Stats":
        return f"Words: {word_count} | Characters: {char_count}"
    elif mode == "Uppercase":
        return text.upper()

# Build the UI
demo = gr.Interface(
    fn=analyze_text,
    inputs=[
        gr.Textbox(lines=4, label="Input Prompt / Document", placeholder="Type your text here..."),
        gr.Radio(["Summary", "Word Stats", "Uppercase"], label="Task Mode", value="Summary"),
    ],
    outputs=gr.Textbox(label="Model Output", lines=3),
    title="AI Text Processor",
    description="A lightweight Gradio interface for quick AI prototyping.",
)

if __name__ == "__main__":
    demo.launch()       #share=True
