import json
import os
from services import bedrock_agent_runtime
import streamlit as st
import uuid
# import utils.func
import utils.model_config
from app4_front_end import retrieve_knowledge
from app6_hyperlink_frontend import retrieve_knowledge_hyper

# Get config from environment variables
agent_id = os.environ.get("BEDROCK_AGENT_ID", "XXXXXX") #
agent_alias_id = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "TSTALIASID") # TSTALIASID is the default test alias ID
ui_title = os.environ.get("BEDROCK_AGENT_TEST_UI_TITLE", "GenAI Hub ChatBot - Prompt Pioneers")
ui_icon = os.environ.get("BEDROCK_AGENT_TEST_UI_ICON")

def init_state():
   st.session_state.session_id = str(uuid.uuid4())
   st.session_state.messages = []
   st.session_state.citations = []
   st.session_state.trace = {}

# General page configuration and initialization
st.set_page_config(page_title=ui_title, page_icon=ui_icon, layout="wide")
st.title(ui_title)
if len(st.session_state.items()) == 0:
   init_state()

# Messages in the conversation
for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"], unsafe_allow_html=True)

# Chat input that invokes the agent
if prompt := st.chat_input():
   st.session_state.messages.append({"role": "user", "content": prompt})
   with st.chat_message("user"):
       st.write(prompt)

   with st.chat_message("assistant"):
       placeholder = st.empty()
       placeholder.markdown("...")

       with st.spinner("Retrieving results from the knowledge base..."):
           # results = app4_front_end.retrieve_knowledge(prompt)
           st.write_stream(retrieve_knowledge_hyper(prompt))
           # search_results = ""
           # for result in results:
           #    search_results += result['content']['text'] + '\n\n'

       # st.markdown("✅ Relevant document chunks received")
       #st.write(search_results)

       # input_text = prompt + '\n' + f"search results: {search_results}"
       # message = {
       #    "role": "user",
       #    "content": [{
       #        "text": input_text
       #    }]
       # }
       # messages = [message]

       # with st.spinner("Streaming response from Converse API..."):
           # st.write_stream(
           #    utils.func.stream_bedrock_response(utils.model_config.MODEL_ID, messages, utils.model_config.RAG_SYSTEM_PROMPT, utils.model_config.INFERENCE_CONFIG)
           # )
           # st.write_stream(
           #    app4_front_end.retrieve_knowledge(prompt)
           #    )
           # st.write(app4_front_end.retrieve_knowledge(prompt))
####  

       # st.markdown("------------")
       # st.subheader("Search results:")
       # st.write(search_results)

       # st.session_state.messages.append({"role": "user", "content": prompt})
       # with st.chat_message("user"):
       #    st.write(prompt)

       # with st.chat_message("assistant"):
       #    placeholder = st.empty()
       #    placeholder.markdown("...")


       # response = bedrock_agent_runtime.invoke_agent(
       #    agent_id,
       #    agent_alias_id,
       #    st.session_state.session_id,
       #    prompt
       # )
       # output_text = response["output_text"]

       # # Add citations
       # if len(response["citations"]) > 0:
       #    citation_num = 1
       #    num_citation_chars = 0
       #    citation_locs = ""
       #    for citation in response["citations"]:
       #        end_span = citation["generatedResponsePart"]["textResponsePart"]["span"]["end"] + 1
       #        for retrieved_ref in citation["retrievedReferences"]:
       #            citation_marker = f"[{citation_num}]"
       #            output_text = output_text[:end_span + num_citation_chars] + citation_marker + output_text[end_span + num_citation_chars:]
       #            citation_locs = citation_locs + "\n<br>" + citation_marker + " " + retrieved_ref["location"]["s3Location"]["uri"]
       #            citation_num = citation_num + 1
       #            num_citation_chars = num_citation_chars + len(citation_marker)
       #        output_text = output_text[:end_span + num_citation_chars] + "\n" + output_text[end_span + num_citation_chars:]
       #        num_citation_chars = num_citation_chars + 1
       #    output_text = output_text + "\n" + citation_locs

       # placeholder.markdown(messages, unsafe_allow_html=True)
       # st.session_state.messages.append({"role": "assistant", "content": output_text})
       # st.session_state.citations = response["citations"]
       # st.session_state.trace = response["trace"]

# trace_type_headers = {
#    "preProcessingTrace": "Pre-Processing",
#    "orchestrationTrace": "Orchestration",
#    "postProcessingTrace": "Post-Processing"
# }
# trace_info_types = ["invocationInput", "modelInvocationInput", "modelInvocationOutput", "observation", "rationale"]
