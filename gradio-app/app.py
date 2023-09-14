import torch

import gradio as gr
from transformers import pipeline
from transformers.pipelines.audio_utils import ffmpeg_read
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate

import warnings
warnings.filterwarnings("ignore")

import os
import textwrap

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
from InstructorEmbedding import INSTRUCTOR

import tempfile
import os
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
hugging_face_model = "eachadea/vicuna-7b-1.1"

tokenizer = AutoTokenizer.from_pretrained(hugging_face_model)

llm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", #meta-llama/Llama-2-13b-chat-hf
                                                     load_in_8bit=True,
                                                     device_map='balanced_low_0',
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     token=access_token
                                                    )
max_len = 4096
llm_task = "text-generation"
T = 0

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
template = """You are a helpful AI assistant and provide a detailed summary for the given context which are meeting notes. Divide your response into a Summary, Key takeaways and Action Items.
Context:{context}
>>Summary<<"""
prompt_template = PromptTemplate(input_variables=["context"], template = template)


text_chain = LLMChain(llm=text_llm, prompt=prompt_template)





# def chain(query, retriever):
#     """
#     Executes a retrieval-based question-answering chain with specified query and retriever.

#     Args:
#     - query (str): The query/question to be answered.
#     - retriever (Retriever): The retriever object responsible for fetching relevant documents.

#     Returns:
#     - dict: Response from the RetrievalQA.
#     """
#     qa_chain = RetrievalQA.from_chain_type(llm=llm, 
#                                        chain_type="stuff", 
#                                        retriever=set_retriver(retriever), 
#                                        return_source_documents=True,
#                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
#                                        verbose=True)
#     return qa_chain(query)


##Updated Code###

def transcribe(inputs, task):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    ##Updated Code###
    sum_text = text_chain(text)
    ##Updated Code###
    print(sum_text["text"])
    return  sum_text['context'], sum_text["text"]


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs="text",
    layout="horizontal",
    theme="huggingface",
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="upload", type="filepath", optional=True, label="Audio file"),
        gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs=[
        gr.outputs.Textbox(label="Transcribed Text using Whisper"),
        gr.outputs.Textbox(label="Summarized Text using Llama-2")],
    layout="horizontal",
    theme="huggingface",
    allow_flagging="never",
)


with demo:
    gr.TabbedInterface([file_transcribe, mf_transcribe], ["Audio file","Microphone"])

if __name__ == "__main__":
    demo.launch(share=True,
                enable_queue=True,
                show_error=True,
                server_name='127.0.0.1',
                server_port=int(os.getenv('CDSW_APP_PORT'))) 

    print("Gradio app ready")