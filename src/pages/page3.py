import streamlit as st
import boto3
import uuid
import time
import os

# AWS認証情報の設定確認
required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}")
else:
    # Bedrockクライアントの初期化
    if "client" not in st.session_state:
        st.session_state.client = boto3.client("bedrock-agent-runtime")
    client = st.session_state.client

    # セッションIDの生成
    if "session_id" not in st.session_state:
        timestamp = int(time.time())
        st.session_state.session_id = f"agent-{timestamp}-{uuid.uuid1()}"

    st.title("マルチエージェント相談")

    # 相談内容の入力
    query = st.text_area("相談内容を入力してください：", height=100)

    if st.button("相談する"):
        prompt = f"""
        Please provide advice on the following query:
        {query}
        
        Use a single project table in Working Memory with table name: {st.session_state.session_id}
        """
        
        response = client.invoke_agent(
            agentId="LR5H7Z5ZDD",
            agentAliasId="V12WREIOGZ",
            sessionId=st.session_state.session_id,
            enableTrace=True,
            inputText=prompt
        )
        orch_step = 0
        sub_step = 0
        action_output = {"flag": False, "content": ""}

        # multi_agent_namesのキーは、各サブエージェントの "{エージェントID}/{エイリアスID}"
        multi_agent_names = {
        "XMF9QM0BUV/UYB938TOTJ": "INTP", 
        "KOSY0CSVGJ/TPMVQZ74WJ" : "ENFJ",
        }

        with st.chat_message("assistant"):
            for event in response.get("completion"):
                sub_agent_alias_id = None

                if "trace" in event:
                    if "callerChain" in event["trace"]:
                        if len(event["trace"]["callerChain"]) > 1:
                            sub_agent_alias_arn = event["trace"]["callerChain"][1]["agentAliasArn"]
                            sub_agent_alias_id = sub_agent_alias_arn.split("/", 1)[1]
                            sub_agent_name = multi_agent_names[sub_agent_alias_id]

                    if "orchestrationTrace" in event["trace"]["trace"]:
                        orch = event["trace"]["trace"]["orchestrationTrace"]

                        if "invocationInput" in orch:
                            _input = orch["invocationInput"]
                            if "agentCollaboratorInvocationInput" in _input:
                                collab_name = _input["agentCollaboratorInvocationInput"]["agentCollaboratorName"]
                                sub_agent_name = collab_name
                                collab_input_text = _input["agentCollaboratorInvocationInput"]["input"]["text"]
                                collab_arn = _input["agentCollaboratorInvocationInput"]["agentCollaboratorAliasArn"]
                                collab_ids = collab_arn.split("/", 1)[1]

                                action_output = {
                                    "flag": False,
                                    "content": f"Using sub-agent collaborator: '{collab_name} [{collab_ids}]'"
                                }

                        if "observation" in orch:
                            output = orch["observation"]
                            if "actionGroupInvocationOutput" in output:
                                action_output = {
                                    "flag": True,
                                    "content": f"--tool outputs:\n{output['actionGroupInvocationOutput']['text']}"
                                }

                            if "agentCollaboratorInvocationOutput" in output:
                                collab_name = output["agentCollaboratorInvocationOutput"]["agentCollaboratorName"]
                                collab_output_text = output["agentCollaboratorInvocationOutput"]["output"]["text"]
                                action_output = {
                                    "flag": True,
                                    "content": f"----sub-agent {collab_name} output text:\n{collab_output_text}"
                                }

                        if "modelInvocationOutput" in orch:
                            if action_output["flag"] and len(action_output["content"]) > 0:
                                sub_step += 1
                                with st.expander(f"---- Step {orch_step}.{sub_step} [using sub-agent name:{sub_agent_name}]"):
                                    st.write(action_output["content"])
                            elif (not action_output["flag"]) and len(action_output["content"]) > 0:
                                orch_step += 1
                                sub_step = 0
                                with st.expander(f"---- Step {orch_step} ----"):
                                    st.write(action_output["content"])

                if "chunk" in event:
                    answer = event["chunk"]["bytes"].decode()
                    st.write(answer)