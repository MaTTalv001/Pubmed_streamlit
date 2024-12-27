import streamlit as st
import boto3
from langchain_community.chat_models import BedrockChat
import os

def initialize_bedrock():
    required_env_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(f"必要な環境変数が見つかりません: {', '.join(missing_vars)}")
    
    bedrock = boto3.client('bedrock-runtime')
    llm = BedrockChat(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        client=bedrock
    )
    return llm

# MBTI性格タイプの定義
MBTI_TYPES = {
    'INTJ': {'name': '建築家', 'description': '論理的で戦略的な思考を持ち、独創的なビジョンを持つタイプ'},
    'INTP': {'name': '論理学者', 'description': '革新的で好奇心旺盛なタイプ'},
    'ENTJ': {'name': '指揮官', 'description': '大胆で想像力豊かなリーダー気質のタイプ'},
    'ENTP': {'name': '討論者', 'description': '知的好奇心が強く、論争を好むタイプ'},
    'INFJ': {'name': '提唱者', 'description': '内向的で洞察力があり、理想主義的なタイプ'},
    'INFP': {'name': '仲介者', 'description': '理想主義的で献身的なタイプ'},
    'ENFJ': {'name': '主人公', 'description': 'カリスマ的で人々を導くタイプ'},
    'ENFP': {'name': '広報運動家', 'description': '情熱的で創造的なタイプ'},
    'ISTJ': {'name': '管理者', 'description': '実践的で事実重視のタイプ'},
    'ISFJ': {'name': '擁護者', 'description': '献身的で温かい心の持ち主タイプ'},
    'ESTJ': {'name': '幹部', 'description': '実務的で伝統を重んじるタイプ'},
    'ESFJ': {'name': '領事官', 'description': '世話好きで社交的なタイプ'},
    'ISTP': {'name': '巨匠', 'description': '大胆で実践的なタイプ'},
    'ISFP': {'name': '冒険家', 'description': '柔軟で魅力的なタイプ'},
    'ESTP': {'name': '起業家', 'description': '活発で現実的なタイプ'},
    'ESFP': {'name': 'エンターテイナー', 'description': '自発的で活発なタイプ'}
}

def generate_mbti_response(llm, query, personality_type):
    prompt = f"""以下の相談内容に対して、{personality_type}タイプ（{MBTI_TYPES[personality_type]['name']}）の視点から回答してください。
回答は、その性格タイプの特徴を反映した、現実的でありながらその性格らしいアドバイスにしてください。

相談内容：
{query}

要件：
- その性格タイプらしい特徴的な考え方や価値観を反映させる
- 実践的で具体的なアドバイスを含める
- 100字程度で簡潔に回答する
"""
    response = llm.predict(prompt)
    return response

def generate_discussion(llm, query, selected_types):
    personalities = [f"{type}（{MBTI_TYPES[type]['name']}）" for type in selected_types]
    prompt = f"""以下の相談内容について、選ばれた性格タイプ同士で建設的な議論を展開してください。
各発言は必ず「{personalities[0]}:」「{personalities[1]}:」のように発言者を明示してください。
最後は必ず「■まとめ」という見出しで議論の要点をまとめてください。

相談内容：
{query}

参加する性格タイプ：
{', '.join(personalities)}

要件：
- 各性格タイプの特徴を活かした視点で議論を展開
- お互いの意見を尊重しながら、建設的な議論を行う
- 3-4往復程度の自然な会話形式で表現
- 各発言は簡潔に50字程度で
- 最後に「■まとめ」として議論の要点をまとめる
"""
    response = llm.predict(prompt)
    messages = []
    summary = ""
    
    lines = response.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith('■まとめ'):
                summary = line.replace('■まとめ', '').strip()
            elif ':' in line:
                messages.append(line.strip())
    
    return messages, summary

def main():
    st.title("MBTI性格タイプ別アドバイザー")
    
    # セッション状態の初期化
    if 'responses' not in st.session_state:
        st.session_state.responses = {}
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'selected_types' not in st.session_state:
        st.session_state.selected_types = []
    if 'discussion_generated' not in st.session_state:
        st.session_state.discussion_generated = False
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'comment_submitted' not in st.session_state:
        st.session_state.comment_submitted = False

    try:
        llm = initialize_bedrock()
    except Exception as e:
        st.error("Bedrock初期化エラー: " + str(e))
        return

    # 相談内容入力エリア
    query = st.text_area("相談内容を入力してください：", height=100)
    
    # 性格タイプ選択エリア
    st.write("分析に使用する性格タイプを選択してください（最大3つ）")
    
    cols = st.columns(3)
    selected_types = []
    
    for i, (mbti_type, info) in enumerate(MBTI_TYPES.items()):
        with cols[i % 3]:
            if st.checkbox(f"{mbti_type} ({info['name']})", 
                         help=info['description'],
                         key=f"checkbox_{mbti_type}"):
                selected_types.append(mbti_type)
    
    if len(selected_types) > 3:
        st.warning("性格タイプは最大3つまで選択できます。")
        selected_types = selected_types[:3]

    response_cols = st.columns(3)

    if st.button("回答を生成") and query and selected_types:
        st.session_state.query = query
        st.session_state.selected_types = selected_types
        st.session_state.responses = {}
        
        for i, mbti_type in enumerate(selected_types):
            with response_cols[i]:
                st.subheader(f"{mbti_type}（{MBTI_TYPES[mbti_type]['name']}）の回答")
                with st.spinner("回答を生成中..."):
                    try:
                        response = generate_mbti_response(llm, query, mbti_type)
                        st.session_state.responses[mbti_type] = response
                        st.write(response)
                    except Exception as e:
                        st.error(f"回答生成エラー: {str(e)}")

    else:
        for i, mbti_type in enumerate(st.session_state.selected_types):
            if i < 3:
                with response_cols[i]:
                    if mbti_type in st.session_state.responses:
                        st.subheader(f"{mbti_type}（{MBTI_TYPES[mbti_type]['name']}）の回答")
                        st.write(st.session_state.responses[mbti_type])

    if len(selected_types) >= 2 and bool(st.session_state.responses):
        col1, col2 = st.columns([1, 4])  # カラムを作成してボタンを左側に配置
        
        with col1:
            if st.button("ディスカッションを生成"):
                with st.spinner("議論を生成中..."):
                    try:
                        messages, summary = generate_discussion(llm, query, selected_types)
                        st.session_state.discussion_messages = messages
                        st.session_state.discussion_summary = summary
                        st.session_state.discussion_generated = True
                    except Exception as e:
                        st.error(f"議論生成エラー: {str(e)}")
        
        # ディスカッションが生成されている場合は常に表示
        with col2:
            if st.session_state.discussion_generated:
                st.markdown("**ディスカッションが生成されました**")
        
        # ディスカッションの内容表示（ボタンの下に配置）
        if st.session_state.discussion_generated:
            st.subheader("性格タイプ間の議論")
            
            # 元の相談内容を表示
            st.markdown("**元の相談内容：**")
            st.info(st.session_state.query)
            
            # 最初のディスカッションを表示
            for message in st.session_state.discussion_messages:
                parts = message.split(':', 1)
                if len(parts) == 2:
                    speaker, content = parts
                    speaker = speaker.strip()
                    content = content.strip()
                    
                    speaker_type = speaker.split('（')[0]
                    speaker_name = MBTI_TYPES[speaker_type]['name']
                    
                    is_user = speaker_type == selected_types[0]
                    
                    with st.chat_message("user" if is_user else "assistant"):
                        st.markdown(f"**{speaker_type}（{speaker_name}）**")
                        st.write(content)
            
            # 最初のディスカッションを表示
            for message in st.session_state.discussion_messages:
                parts = message.split(':', 1)
                if len(parts) == 2:
                    speaker, content = parts
                    speaker = speaker.strip()
                    content = content.strip()
                    
                    # MBTI型の取得方法を修正
                    speaker_parts = speaker.split('（')
                    speaker_type = speaker_parts[0].strip()  # 括弧の前の部分（MBTI型）を取得
                    
                    try:
                        speaker_name = MBTI_TYPES[speaker_type]['name']
                    except KeyError:
                        st.error(f"未知の性格タイプ: {speaker_type}")
                        speaker_name = "不明"
                    
                    is_user = speaker_type == selected_types[0]
                    
                    with st.chat_message("user" if is_user else "assistant"):
                        st.markdown(f"**{speaker_type}（{speaker_name}）**")
                        st.write(content)
            
            # 保存された会話履歴を表示
            for history_item in st.session_state.conversation_history:
                if history_item['type'] == 'user':
                    with st.chat_message("user"):
                        st.markdown("**あなた**")
                        st.write(history_item['content'])
                else:
                    with st.chat_message("assistant"):
                        speaker_type = history_item['speaker']
                        try:
                            speaker_name = MBTI_TYPES[speaker_type]['name']
                        except KeyError:
                            st.error(f"未知の性格タイプ: {speaker_type}")
                            speaker_name = "不明"
                        st.markdown(f"**{speaker_type}（{speaker_name}）**")
                        st.write(history_item['content'])
            
            # まとめの表示
            if st.session_state.discussion_summary:
                st.markdown("---")
                st.markdown("### 議論のまとめ")
                st.info(st.session_state.discussion_summary)
            
            # コメント入力部分を最後に配置
            # コメント入力部分を最後に配置
            st.markdown("---")
            
            # コメント送信後に入力欄をクリアするためのキー管理
            if 'comment_key' not in st.session_state:
                st.session_state.comment_key = 0
            
            new_comment = st.text_area(
                "議論に参加するコメントを入力してください：", 
                key=f"new_comment_{st.session_state.comment_key}"
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("コメントを送信"):
                    if new_comment and not st.session_state.comment_submitted:
                        st.session_state.comment_submitted = True
                        # ユーザーのコメントを会話履歴に追加
                        st.session_state.conversation_history.append({
                            'type': 'user',
                            'content': new_comment
                        })
                        
                        # 選択された性格タイプからの応答を生成
                        with st.spinner("返信を生成中..."):
                            for mbti_type in selected_types:
                                response_prompt = f"""元の相談内容と、これまでの議論を踏まえて回答してください。

元の相談内容：
{st.session_state.query}

新しいコメント：
{new_comment}

あなたは{mbti_type}（{MBTI_TYPES[mbti_type]['name']}）として、この文脈で応答してください。

要件：
- その性格タイプらしい考え方や価値観を反映させる
- これまでの議論の文脈を考慮する
- 元の相談内容を踏まえた回答をする
- 50字程度で簡潔に応答する"""

                                try:
                                    response = llm.predict(response_prompt)
                                    st.session_state.conversation_history.append({
                                        'type': 'assistant',
                                        'speaker': mbti_type,
                                        'content': response
                                    })
                                except Exception as e:
                                    st.error(f"応答生成エラー ({mbti_type}): {str(e)}")
                        
                        # 入力欄をクリアするためにキーを更新
                        st.session_state.comment_key += 1
                        # 画面を更新
                        st.rerun()
            
            with col2:
                if st.session_state.comment_submitted:
                    st.success("コメントが送信され、返信が生成されました")

            # 送信状態をリセット
            if st.session_state.comment_submitted:
                st.session_state.comment_submitted = False

if __name__ == "__main__":
    main()