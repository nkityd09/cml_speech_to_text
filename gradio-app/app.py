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




# prompt_template = """Write a three part report of the following Text divided into Detailed Summary, Key Takeaways and Action Items:


# "{text}"


# CONCISE SUMMARY:"""
# PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

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

summary_chain = load_summarize_chain(llm=text_llm, chain_type="map_reduce", return_intermediate_steps=True , map_prompt=MAP_PROMPT,combine_prompt=COMBINE_PROMPT,token_max=8192 ,verbose=True)



def transcribe(inputs, task):
    if inputs is None:
        raise gr.Error("No audio file submitted! Please upload or record an audio file before submitting your request.")

    text = pipe(inputs, batch_size=BATCH_SIZE, generate_kwargs={"task": task}, return_timestamps=True)["text"]
    ##Updated Code###

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    output = summary_chain(docs)
    summary = output["output_text"]
    return text, summary


demo = gr.Blocks()

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath", optional=True),
        gr.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
        #gr.inputs.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
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
        gr.Radio(["transcribe", "translate"], label="Task", default="transcribe"),
    ],
    outputs=[gr.Textbox(label="Transcribed Text using Whisper"), gr.Textbox(label="Summarized Text using Llama-2")],
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