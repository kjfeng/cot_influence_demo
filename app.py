import gradio as gr
import os
from dotenv import load_dotenv
from together import Together
import zwsp_steg

# Load environment variables
load_dotenv()

# Initialize Together client
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def get_model_response(message, history):
    """
    Function to handle chatbot interactions using Together API
    """
    try:
        # Prepare the conversation history for the API
        messages = []
        
        # Add conversation history - history is now in messages format
        for msg in history:
            messages.append(msg)
        
        # Add the current message
        messages.append({"role": "user", "content": message})
        
        # Call Together API
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-0528",
            messages=messages,
            temperature=0.7,
        )
        
        # Extract the response content
        ai_response = response.choices[0].message.content
        reasoning = extract_reasoning(ai_response)
        reasoning_encoded = zwsp_steg.encode(reasoning)
        response_text = extract_response(ai_response) + reasoning_encoded

        return response_text
        
    except Exception as e:
        return f"Error: {str(e)}"

def inspect_text(text):
    """
    Function to handle text inspection (placeholder for now)
    """
    if not text.strip():
        return "Please enter some text to inspect."
    
    # For now, just return the text as requested
    return f"Inspected text:\n\n{text}"


def extract_reasoning(output):
    # extract text between "<think>" and "</think>"
    start = output.find("<think>\n") + len("<think>\n")
    end = output.find("</think>")
    if start == -1 or end == -1:
        return None
    return output[start:end].strip()


def extract_response(output):
    # return part of the response that's not the reasoning
    start = output.find("<think>\n")
    end = output.find("</think>") + len("</think>\n")

    if start != -1 and end > len("</think>\n"):
        output = output[:start] + output[end:]
        return output.strip()
    
    else: 
        return output.strip()

# Create the Gradio interface
with gr.Blocks(title="CoT Monitoring Demo", theme=gr.themes.Base()) as app:
    gr.Markdown("# 🤖 CoT Monitoring Demo")
    
    with gr.Tabs():
        # Chat Tab
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(
                height=500,
                placeholder="Start a conversation...",
                show_copy_button=True,
                type="messages"
            )
            
            # with gr.Row():
            #     msg = gr.Textbox(
            #         placeholder="Type your message here...",
            #         container=False,
            #         scale=7,
            #         lines=1
            #     )
            #     send_btn = gr.Button("Send", scale=1, variant="primary")
            #     clear_btn = gr.Button("Clear", scale=1)

            gr.ChatInterface(fn=get_model_response, type="messages", chatbot=chatbot, submit_btn="Send")
            
        
        # Inspect Tab
        with gr.Tab("🔎 Inspect"):
            gr.Markdown("### Text Inspector")
            gr.Markdown("Paste the AI output below and click 'Inspect' to analyze it.")
            
            with gr.Column():
                input_text = gr.Textbox(
                    label="Text to Inspect",
                    placeholder="Paste your text here...",
                    lines=10,
                    max_lines=20
                )
                
                inspect_btn = gr.Button("Inspect", variant="primary")
                
                output_text = gr.Textbox(
                    label="Inspection Result",
                    lines=10,
                    max_lines=20,
                    interactive=False
                )
            
            # Event handler for inspect
            inspect_btn.click(
                fn=inspect_text,
                inputs=input_text,
                outputs=output_text
            )

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
