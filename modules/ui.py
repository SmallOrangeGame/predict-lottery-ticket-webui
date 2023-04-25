import math
import os
import plotly.express as px
import gradio as gr
import gradio.routes
import gradio.utils
import numpy as np

import modules.scripts

plot_end = 2 * math.pi

def create_ui():
    input_textbox = gr.Textbox()

    with gr.Blocks() as demo:
        gr.Examples(["hello", "bonjour", "merhaba"], input_textbox)
        input_textbox.render()

    return demo