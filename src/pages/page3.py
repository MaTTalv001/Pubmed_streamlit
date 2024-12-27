import streamlit as st
import boto3
import uuid
import time
import os

required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}")
else:
    if "client" not in st.session_state:
        st.session_state.client = boto3.client("bedrock-agent-runtime")
    client = st.session_state.client

    if "session_id" not in st.session_state:
        timestamp = int(time.time())
        st.session_state.session_id = f"agent-{timestamp}-{uuid.uuid1()}"

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- チャット表示 ---
    st.title("マルチエージェント相談")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # --- 入力欄を常に下に表示する ---
    # st.chat_input で入力欄を固定表示する
    query = st.chat_input("相談内容を入力してください：")

    # --- 相談送信の処理 ---
    if query:
        # ユーザー入力をチャット履歴に追加して即時表示
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # 履歴からプロンプトを生成
        history = "\n".join([
            f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
            for msg in st.session_state.chat_history[:-1]
        ])
        prompt = f"Previous conversation:\n{history}\n\nCurrent query:\n{query}\nUse Working Memory table: {st.session_state.session_id}"

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
        full_response = ""
        multi_agent_names = {
            "XMF9QM0BUV/UYB938TOTJ": "INTP", 
            "KOSY0CSVGJ/TPMVQZ74WJ": "ENFJ"
        }

        with st.chat_message("assistant"):
            for event in response.get("completion"):
                if "trace" in event:
                    if ("callerChain" in event["trace"]
                        and len(event["trace"]["callerChain"]) > 1):
                        sub_agent_alias_id = event["trace"]["callerChain"][1]["agentAliasArn"].split("/", 1)[1]
                        sub_agent_name = multi_agent_names.get(sub_agent_alias_id, "Unknown")

                    if "orchestrationTrace" in event["trace"]["trace"]:
                        orch = event["trace"]["trace"]["orchestrationTrace"]

                        if ("invocationInput" in orch
                                and "agentCollaboratorInvocationInput" in orch["invocationInput"]):
                            input_data = orch["invocationInput"]["agentCollaboratorInvocationInput"]
                            action_output = {
                                "flag": False,
                                "content": f"Using: '{input_data['agentCollaboratorName']} [{input_data['agentCollaboratorAliasArn'].split('/', 1)[1]}]'"
                            }

                        if "observation" in orch:
                            output = orch["observation"]
                            if "actionGroupInvocationOutput" in output:
                                action_output = {
                                    "flag": True,
                                    "content": f"Tool output:\n{output['actionGroupInvocationOutput']['text']}"
                                }
                            elif "agentCollaboratorInvocationOutput" in output:
                                collab_data = output["agentCollaboratorInvocationOutput"]
                                action_output = {
                                    "flag": True,
                                    "content": f"Agent {collab_data['agentCollaboratorName']} output:\n{collab_data['output']['text']}"
                                }

                        if "modelInvocationOutput" in orch:
                            # アクションの出力を Expander として表示
                            if action_output["flag"] and action_output["content"]:
                                sub_step += 1
                                with st.expander(f"Step {orch_step}.{sub_step} [{sub_agent_name}]"):
                                    st.write(action_output["content"])
                            elif action_output["content"]:
                                orch_step += 1
                                sub_step = 0
                                with st.expander(f"Step {orch_step}"):
                                    st.write(action_output["content"])

                if "chunk" in event:
                    chunk_text = event["chunk"]["bytes"].decode()
                    full_response += chunk_text
                    st.write(chunk_text)

            if full_response:
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
