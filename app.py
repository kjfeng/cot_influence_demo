import gradio as gr
from utils import *

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

            reasoning_accordion = gr.Accordion("Show reasoning", open=False, visible=False, render=False)
            with reasoning_accordion:
                # reasoning_chain = gr.Textbox(
                #     label="Encoded reasoning chain",
                #     lines=20,
                #     max_lines=100,
                #     interactive=False,
                # )

                reasoning_chain = gr.Markdown()


            gr.ChatInterface(
                fn=get_model_response, 
                type="messages", 
                chatbot=chatbot, 
                submit_btn="Send", 
                examples=examples_strs, 
                example_labels=example_labels,
                run_examples_on_click=False,
                additional_outputs=[reasoning_chain, reasoning_accordion]
            )

            reasoning_accordion.render()



        # Inspect Tab
        with gr.Tab("🔎 Inspect"):
            gr.Markdown("### Output Inspector")
            
            with gr.Column():
                input_text = gr.Textbox(
                    label="Output to Inspect",
                    placeholder="Paste text here...",
                    lines=10
                )
                
                inspect_btn = gr.Button("Inspect", variant="primary")
                
                decision_output = gr.Textbox(
                    label="Decision",
                    lines=1,
                    interactive=False
                )
                
                evidence_accordion = gr.Accordion("Evidence", open=False, visible=False)
                with evidence_accordion:
                    evidence_output = gr.Textbox(
                        label="Evidence Details",
                        lines=10,
                        max_lines=100,
                        interactive=False,
                        elem_classes="reasoning-box"
                    )
                
                reasoning_accordion = gr.Accordion("Show reasoning", open=False, visible=False)
                with reasoning_accordion:
                    reasoning_output = gr.Markdown()
            
            # Event handler for inspect
            inspect_btn.click(
                fn=format_grader_output,
                inputs=input_text,
                outputs=[decision_output, evidence_output, evidence_accordion, reasoning_output, reasoning_accordion]
            )

# Launch the app
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
