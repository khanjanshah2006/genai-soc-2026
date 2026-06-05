import gradio as gr
from indexing import index_documents
from retrieval import ask

def process_files(uploaded_files):
    if not uploaded_files:
        return "No files selected."
    
    total_chunks = index_documents(uploaded_files)
    num_docs = len(uploaded_files)
    
    return f"{num_docs} documents indexed — {total_chunks} total chunks."

def chat(user_message, history):
    if not user_message.strip():
        return "", history, "No context retrieved."

    answer, context_display = ask(user_message)
    
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    
    return "", history, context_display

with gr.Blocks(title="DocBuddy RAG") as demo:
    gr.Markdown("DocBuddy - A RAG Application")
    
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Conversation")
            msg_input = gr.Textbox(
                placeholder="Ask a question about your documents...", 
                label="Your Question"
            )
            
            with gr.Accordion("Retrieved Context", open=False):
                context_box = gr.Markdown("*Upload a document and ask a question to see the AI's reading material here.*")
        
        with gr.Column(scale=1):
            file_upload = gr.File(
                file_count="multiple", 
                file_types=[".pdf"], 
                label="1. Upload PDFs",
                type="filepath" 
            )
            index_btn = gr.Button("2. Index Documents", variant="primary")
            
            status_label = gr.Textbox(
                label="System Status", 
                value="No documents indexed yet", 
                interactive=False
            )

    index_btn.click(
        fn=process_files,
        inputs=[file_upload],
        outputs=[status_label]
    )
    
    msg_input.submit(
        fn=chat,
        inputs=[msg_input, chatbot],
        outputs=[msg_input, chatbot, context_box]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())