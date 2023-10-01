import torch
from langchain.prompts import PromptTemplate
import gradio as gr
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
import warnings
warnings.filterwarnings("ignore")
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import textwrap
import langchain
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from InstructorEmbedding import INSTRUCTOR
langchain.verbose = True
import tempfile
import os
import pandas as pd
from datetime import datetime
print(os.environ['PATH'])

MODEL_NAME = "openai/whisper-large-v2"
BATCH_SIZE = 8
FILE_LIMIT_MB = 1000

access_token = os.environ["HF_TOKEN"]
!huggingface-cli login --token $HF_TOKEN


device = 0 if torch.cuda.is_available() else "cpu"

pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)



## Updated Code
hugging_face_model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)

llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", #meta-llama/Llama-2-13b-chat-hf
                                                     load_in_4bit=True,
                                                     device_map='balanced_low_0',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     token=access_token
                                                    )
max_len = 8192
llm_task = "text-generation"
T = 0.1

llm_pipeline = pipeline(
    task=llm_task,
    model=llm_model, 
    tokenizer=tokenizer, 
    max_length=max_len,
    temperature=T,
    top_p=0.95,
    repetition_penalty=1.15
)

text_llm = HuggingFacePipeline(pipeline=llm_pipeline)

# Prompt Template for Langchain
template = """You are a helpful AI assistant, please provide a detailed summary for the given context which are meeting notes. Divide your response into a Summary, Key takeaways and Action Items.
Context:{context}
>>Summary<<"""
prompt_template = PromptTemplate(input_variables=["context"], template = template)


text_chain = LLMChain(llm=text_llm, prompt=prompt_template)


map_prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:"""
MAP_PROMPT = PromptTemplate(template=map_prompt_template, input_variables=["text"])


combine_prompt_template = """You will be given a series of summaries from a meeting. The summaries will be enclosed in triple backticks (```)
Your goal is to give a detailed summary of what happened in the meeting.
The reader should be able to grasp what happened in the meeting.
```{text}```
VERBOSE SUMMARY:"""
COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

# summary_chain = load_summarize_chain(llm=text_llm, chain_type="map_reduce", return_intermediate_steps=True , map_prompt=MAP_PROMPT,combine_prompt=COMBINE_PROMPT,token_max=8192 ,verbose=True)


columns = ['Time', 'Map_Prompt', 'Combine_Prompt', 'Summary']
logging_df = pd.DataFrame(columns=columns)

def logging(df, datetime, map_prompt, combine_prompt, summary):
    new_row = {
        'Time': datetime,
        'Map_Prompt': map_prompt,
        'Combine_Prompt': combine_prompt,
        'Summary': summary
    }
    df.loc[len(df)] = new_row
    df.to_csv("/home/cdsw/prompt_log/prompt_log.csv")
    return df


def transcribe_summarize(inputs, task, m_prompt, c_prompt):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    ##Updated Code###

    MAP_PROMPT = PromptTemplate(template=m_prompt, input_variables=["text"])
    COMBINE_PROMPT = PromptTemplate(template=c_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=text_llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=MAP_PROMPT,combine_prompt=COMBINE_PROMPT,token_max=8192 ,verbose=True)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    
    docs = text_splitter.create_documents([text])
    output = summary_chain(docs)
    summary = output["output_text"]
    log_df = logging(logging_df, datetime.now(), m_prompt, c_prompt, summary)
    return text, summary, log_df


def transcribe(inputs, task):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]

    return text


def summarize(text, m_prompt, c_prompt):
    MAP_PROMPT = PromptTemplate(template=m_prompt, input_variables=["text"])
    COMBINE_PROMPT = PromptTemplate(template=c_prompt, input_variables=["text"])
    summary_chain = load_summarize_chain(llm=text_llm, chain_type="map_reduce", return_intermediate_steps=True , map_prompt=MAP_PROMPT,combine_prompt=COMBINE_PROMPT,token_max=8192 ,verbose=True)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    output = summary_chain(docs)
    summary = output["output_text"]
    log_df = logging(logging_df, datetime.now(), m_prompt, c_prompt, summary)
    return summary, log_df


with gr.Blocks() as demo:
    gr.Markdown("# Whisper Speech to Text")
    with gr.Row():
        audio_file = gr.Audio(source="upload", type="filepath", label="Audio file")
        transcribed_text = gr.Textbox(label="Text from Whisper", show_copy_button=True)
    with gr.Row():
        task = gr.Radio(["transcribe", "translate"], default="transcribe", label="Task")
        transcribe_button = gr.Button("Transcribe")
        ts_button = gr.Button("Transcribe & Summarize")
    gr.Markdown("# ")
    gr.Markdown("# Llama-2 ")
    with gr.Row():
        with gr.Column(scale=1):
            summary_button = gr.Button("Generate Summary")
        with gr.Column(scale=3):
            summarized_text = gr.Textbox(label="Summarized Text", show_copy_button=True)
    with gr.Accordion("Advanced Options", open=False):
        with gr.Row():
            map_prompt_ui = gr.Textbox(label="Map Prompt", interactive=True, show_copy_button=True, value=map_prompt_template, lines=5)
            combine_prompt_ui = gr.Textbox(label="Combine Prompt", interactive=True, show_copy_button=True, value=combine_prompt_template, lines=5)
    with gr.Accordion("Prompt Log", open=False):
        log_df_ui = gr.DataFrame(wrap=True)
        
    transcribe_button_submit = transcribe_button.click(fn=transcribe, inputs=[audio_file, task], outputs=[transcribed_text])
    summarize_button_submit = summary_button.click(fn=summarize, inputs=[transcribed_text, map_prompt_ui, combine_prompt_ui ], outputs=[summarized_text, log_df_ui])
    all_button_submit = ts_button.click(fn=transcribe_summarize, inputs=[audio_file, task, map_prompt_ui, combine_prompt_ui], outputs=[transcribed_text, summarized_text, log_df_ui])

if __name__ == "__main__":
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT'))) 

    print("Gradio app ready")