# CML Speech To Text
AMP demonstrating Speech to Text Summarization on CML using OpenAI Whisper

## Key Features
- Speech to Text transcription using OSS Whisper Model
- Summarize transcribed Text using Llama-2 (can be changed during AMP deployment) and LangChain
- Customizable Prompts for various summarization use cases
- Prompt Log for monitoring and tracking how different prompts perform. Log gets stored in CSV which can be used as data for fine tuning downstream.


## Prerequistes
- [Add CML Runtime with required packages for Speech to Text](https://github.com/nkityd09/cml_speech_to_text/blob/main/prerequistes/CML_Runtime.md)

## Resource Requirements
The AMP Application has been configured to use the following
- 4 CPU
- 32 GB RAM
- 2 GPUs

## Steps to Setup the CML Application

1. Navigate to CML Workspace -> Site Administration -> AMPs Tab

2. Under AMP Catalog Sources section, We will "Add a new source By" selecting "Catalog File URL"

3. Provide the following URL and click "Add Source"
```
https://raw.githubusercontent.com/nkityd09/cml_speech_to_text/main/catalog.yaml
```

4. Once added, We will be able to see the LLM PDF Document Chatbot in the AMP section and deploy it from there.

5. Click on the AMP and "Configure Project"
   - Add HuggingFace Model Name, defaults to meta-llama/Llama-2-7b-chat-hf
   - If accessing a gated model, add HuggingFace token. Can be left blank for non gated models
[Configuring AMP](images/configuring_amp.png)

6. Once the AMP steps are completed, We can access the Gradio UI via the Applications page.

## App In Action

The CML Application serves a Gradio UI to upload Audio files and summarize the text transcribed from the Audio.

1. The Gradio UI provides an upload widget which can be used to upload Audio files
   [Application UI](images/application_ui.png)
2. Once the file has been uploaded, we can either Transcribe the audio file or Transcribe and Summarize its text.
   [Application run](images/application_run.png)
3. The default prompts are set for summarizing meeting notes but can be changed from the **Advanced Options** section
   [Advanced Options](images/advanced_options.png)



