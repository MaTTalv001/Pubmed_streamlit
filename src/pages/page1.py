import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from main import fetch_pubmed_articles, is_valid_email

def create_keyword_network(articles_df):
   nodes = []
   edges = []
   keyword_freq = {}
   keyword_pairs = {}
   
   for _, paper in articles_df.iterrows():
       keywords = [kw.lower() for kw in paper['Keywords'].split('; ')]
       for kw in keywords:
           if kw:  # 空文字列でない場合のみ
               keyword_freq[kw] = keyword_freq.get(kw, 0) + 1
               
       # 共起関係をカウント
       for i, kw1 in enumerate(keywords):
           for kw2 in keywords[i+1:]:
               if kw1 and kw2:  # 両方とも空文字列でない場合
                   pair = tuple(sorted([kw1, kw2]))
                   keyword_pairs[pair] = keyword_pairs.get(pair, 0) + 1

   # ノード作成
   for kw, freq in keyword_freq.items():
       nodes.append(Node(
           id=kw,
           label=f"{kw[:20]}\n({freq})",
           size=2 + (freq * 5),
           color="#00FF00"  # ノードの色
       ))

   # エッジ作成
   for (kw1, kw2), weight in keyword_pairs.items():
       edges.append(Edge(
           source=kw1,
           target=kw2,
           label=f"{weight}",
           width=weight  # 共起回数に応じて線の太さを変える
       ))

   config = Config(
       width=800,
       height=600,
       directed=False,
       physics=True,
       hierarchical=False,
       nodeHighlightBehavior=True,
       highlightColor="#F7A7A6",
       font={'size': 3}
   )
   
   return nodes, edges, config

def main():
   st.title("研究キーワードネットワーク")
   
   email = st.text_input("メールアドレス:")
   
   num_pmids = st.number_input("入力するPMID数:", min_value=1, max_value=10, value=3)
   
   pmids = []
   cols = st.columns(5)  # 2行5列で表示
   for i in range(num_pmids):
       col_idx = i % 5
       with cols[col_idx]:
           pmid = st.text_input(f"PMID {i+1}:")
           pmids.append(pmid)

   if st.button("キーワードネットワークを生成"):
       if not is_valid_email(email):
           st.error("有効なメールアドレスを入力してください")
           return
           
       valid_pmids = [pid.strip() for pid in pmids if pid.strip()]
       
       if not valid_pmids:
           st.error("少なくとも1つのPMIDを入力してください")
           return

       with st.spinner('論文情報を取得中...'):
           df = fetch_pubmed_articles(valid_pmids, email)
           
           if not df.empty:
               st.subheader("キーワードネットワーク")
               nodes, edges, config = create_keyword_network(df)
               agraph(nodes=nodes, edges=edges, config=config)
               
               with st.expander("論文情報"):
                   st.dataframe(df[['Title', 'Authors', 'Keywords']])
           else:
               st.error("論文情報を取得できませんでした")

if __name__ == "__main__":
   main()