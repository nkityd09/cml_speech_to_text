# CML Speech To Text
AMP demonstrating Speech to Text Summarization on CML using OpenAI Whisper

## Key Features
- Uses OSS OpenAI Whisper Model for Speech to Text 
- Provides a summary of the audio entered
- Model of Choice for summarizing text from HuggingFace

## Prerequistes
- Add CML Runtime with required packages for Speech to Text

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
https://raw.githubusercontent.com/nkityd09/cml_chatbot/main/catalog.yaml
```

4. Once added, We will be able to see the LLM PDF Document Chatbot in the AMP section and deploy it from there.

5. Click on the AMP and "Configure Project", disable Spark as it is not required.

6. Once the AMP steps are completed, We can access the Gradio UI via the Applications page.


## Steps to launch the AMP

## WIP

## App In Action
<img width="1308" alt="Speech_To_Text_App" src="https://github.com/nkityd09/cml_speech_to_text/assets/101132317/1693222c-0cbc-4838-b115-72991fd19a5e">


