import gradio as gr

# Define your retrieval_chain here (replace with actual retrieval chain logic)
def get_response(query):
    # Replace this placeholder with your actual retrieval chain call
    response = "This is a simulated response for: **" + query + "**"
    return response

def gradio_query(query, history):
    # Process the query and add it to the chat history
    try:
        result = get_response(query)
        history.append(("You: " + query, "PBMAssist: " + result))
    except Exception as e:
        history.append(("You: " + query, "PBMAssist: Error processing your query."))
        print(f"Error processing query: {e}")
    return history, ""

def clear_chat():
    # Clears the chat history and query input
    return [], ""

# Custom CSS for CVS Health theme
custom_css = """
body {
    background-color: #f8f8f8;
}
.gradio-container {
    max-width: 800px !important;
    margin: auto !important;
}
.custom-chatbot {
    border: 1px solid #cc0000;
    border-radius: 10px;
    overflow: hidden;
}
.custom-chatbot .message {
    padding: 10px;
    margin: 5px;
    border-radius: 5px;
}
.custom-chatbot .user {
    background-color: #f0f0f0;
}
.custom-chatbot .bot {
    background-color: #ffeeee;
}
.custom-button {
    background-color: #cc0000 !important;
    color: white !important;
}
.custom-button:hover {
    background-color: #990000 !important;
}
.logo-container {
    text-align: center;
    margin-bottom: 20px;
}
.logo-container img {
    max-width: 200px;
    height: auto;
}
"""

# UI Configuration
with gr.Blocks(css=custom_css) as demo:
    # CVS Health logo
    gr.HTML(
        """
        <div class="logo-container">
            <img src="https://logos-world.net/wp-content/uploads/2020/11/CVS-Health-Logo.png" alt="CVS Health Logo">
        </div>
        """
    )
    
    # Welcome message (center-aligned)
    gr.Markdown(
        "<div style='text-align: center;'>"
        "<h1 style='color: #cc0000;'>Welcome to CVS Health PBMAssist 👋</h1>"
        "<p>This assistant is here to help you with RxClaim coding queries.</p>"
        "</div>"
    )
    
    with gr.Column():
        # Chat history component
        chat_history = gr.Chatbot(label="PBMAssist", elem_classes="custom-chatbot")
        # Query input box
        query = gr.Textbox(placeholder="Ask your RxClaim-related question...", show_label=False)
        
        with gr.Row():
            # Process and clear buttons
            btn_enter = gr.Button("Process", elem_classes="custom-button")
            btn_clear = gr.Button("Clear Chat", elem_classes="custom-button")
    
    # Button Click Events
    btn_enter.click(fn=gradio_query, inputs=[query, chat_history], outputs=[chat_history, query])
    btn_clear.click(fn=clear_chat, inputs=None, outputs=[chat_history, query])

# Ensure that the app processes requests sequentially
demo.queue()

# Launch the Gradio interface with a public shareable link
demo.launch(share=True, debug=True, inbrowser=True)
