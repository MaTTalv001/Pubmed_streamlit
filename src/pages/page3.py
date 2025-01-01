import streamlit as st
import boto3
import uuid
import time
import os

# --- 必要な環境変数をリスト化 ---
required_env_vars = [
    'AWS_ACCESS_KEY_ID',
    'AWS_SECRET_ACCESS_KEY',
    'AWS_DEFAULT_REGION',
    'AWS_AGENT_ID',
    'AWS_AGENT_ALIAS_ID'
]

# --- 必要な環境変数がセットされているかチェック ---
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

# --- AWSのBedrockエージェント関連のIDを環境変数から取得 ---
AWS_AGENT_ID = os.environ.get('AWS_AGENT_ID')
AWS_AGENT_ALIAS_ID = os.environ.get('AWS_AGENT_ALIAS_ID')

# --- 環境変数が足りなければエラーを表示 ---
if missing_vars:
    st.error(f"Missing environment variables: {', '.join(missing_vars)}")
else:
    # --- セッションステートにclientがなければ初期化 ---
    #     （boto3のbedrock-agent-runtimeクライアントを作成）
    if "client" not in st.session_state:
        st.session_state.client = boto3.client("bedrock-agent-runtime")
    client = st.session_state.client

    # --- セッションIDが未作成なら、現在時刻＋UUIDで一意に作る ---
    if "session_id" not in st.session_state:
        timestamp = int(time.time())
        st.session_state.session_id = f"agent-{timestamp}-{uuid.uuid1()}"

    # --- チャットの履歴を保持するための変数 ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --- チャット画面のタイトル ---
    st.title("マルチエージェント相談")

    # --- セッションステートにある履歴を順番に表示 ---
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # --- チャット入力を常に下に表示するためのUI ---
    query = st.chat_input("相談内容を入力してください：")

    # --- ユーザーがチャット入力を送信した場合の処理 ---
    if query:
        # ユーザー入力をチャット履歴に即時追加
        st.session_state.chat_history.append({"role": "user", "content": query})

        # ユーザーメッセージを即時表示
        with st.chat_message("user"):
            st.write(query)

        # --- 履歴からプロンプトを生成 ---
        #     「Previous conversation:」でこれまでのやり取りをまとめ、
        #     「Current query:」で今回の入力を指定する
        history = "\n".join([
            f"{'User' if msg['role']=='user' else 'Assistant'}: {msg['content']}"
            for msg in st.session_state.chat_history[:-1]
        ])
        prompt = (
            f"Previous conversation:\n{history}\n\n"
            f"Current query:\n{query}\n"
            f"Use Working Memory table: {st.session_state.session_id}"
        )

        # --- Bedrockエージェントへ問い合わせを行う ---
        response = client.invoke_agent(
            agentId=AWS_AGENT_ID,
            agentAliasId=AWS_AGENT_ALIAS_ID,
            sessionId=st.session_state.session_id,
            enableTrace=True,
            inputText=prompt
        )

        # --- トレース表示用に使うカウンタ類 ---
        orch_step = 0
        sub_step = 0

        # アクション出力の管理用フラグ
        action_output = {"flag": False, "content": ""}

        # レスポンス本文を蓄積する変数
        full_response = ""

        # --- エージェント複数呼び出し時にエージェント名をわかりやすくするための辞書 ---
        multi_agent_names = {
            "XMF9QM0BUV/UYB938TOTJ": "ENFJ",
            "KOSY0CSVGJ/TPMVQZ74WJ": "INTP"
        }

        # --- Thinking... 中のスピナー（読み込み中UI） ---
        with st.spinner('Thinking...'):
            # --- アシスタントメッセージ領域に結果を随時書き込む ---
            with st.chat_message("assistant"):
                # invoke_agent から返されるジェネレートされたイベントを順次処理する
                for event in response.get("completion"):
                    # === エージェントのトレース部分を解析 ===
                    if "trace" in event:
                        # サブエージェントの情報があれば取り出す
                        if ("callerChain" in event["trace"]
                            and len(event["trace"]["callerChain"]) > 1):
                            sub_agent_alias_id = event["trace"]["callerChain"][1]["agentAliasArn"].split("/", 1)[1]
                            sub_agent_name = multi_agent_names.get(sub_agent_alias_id, "Unknown")

                        # Orchestrationの詳細トレースを確認
                        if "orchestrationTrace" in event["trace"]["trace"]:
                            orch = event["trace"]["trace"]["orchestrationTrace"]

                            # --- サブエージェントやツール呼び出し前の入力を検知 ---
                            if ("invocationInput" in orch
                                    and "agentCollaboratorInvocationInput" in orch["invocationInput"]):
                                input_data = orch["invocationInput"]["agentCollaboratorInvocationInput"]
                                action_output = {
                                    "flag": False,
                                    "content": f"Using: '[{input_data['agentCollaboratorAliasArn'].split('/', 1)[1]}]'"
                                }

                            # --- サブエージェントやツールからの出力を検知 ---
                            if "observation" in orch:
                                output = orch["observation"]
                                if "actionGroupInvocationOutput" in output:
                                    # ツール（Action）からの出力
                                    action_output = {
                                        "flag": True,
                                        "content": f"Tool output:\n{output['actionGroupInvocationOutput']['text']}"
                                    }
                                elif "agentCollaboratorInvocationOutput" in output:
                                    # サブエージェントからの出力
                                    collab_data = output["agentCollaboratorInvocationOutput"]
                                    action_output = {
                                        "flag": True,
                                        "content": f"Agent {collab_data['agentCollaboratorName']} output:\n{collab_data['output']['text']}"
                                    }

                            # --- メインモデルが出力したテキストを検知 ---
                            if "modelInvocationOutput" in orch:
                                # アクションの出力がある場合はエクスパンダー内に表示
                                if action_output["flag"] and action_output["content"]:
                                    sub_step += 1
                                    with st.expander(f"Opinion:{sub_step}", expanded=False):
                                        st.write(action_output["content"])
                                # まだフラグが立っていない、あるいは別の呼び出しのタイミングの場合
                                elif action_output["content"]:
                                    orch_step += 1
                                    sub_step = 0
                                    # with st.expander(f"Call Agents"):
                                    #     st.write(action_output["content"])

                    # === モデル応答のテキストチャンクを受け取る ===
                    if "chunk" in event:
                        chunk_text = event["chunk"]["bytes"].decode()
                        full_response += chunk_text
                        with st.expander(f"Results", expanded=False):
                            st.write(chunk_text)

                # --- 最後に得られた応答をチャット履歴に追加 ---
                if full_response:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": full_response
                    })
