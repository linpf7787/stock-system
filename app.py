import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from datetime import datetime
import requests
import urllib3
import io

# 隱藏略過 SSL 驗證時產生的警告訊息
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 設定網頁標題與佈局
st.set_page_config(page_title="AI 股票分析與進階指標系統", layout="wide")

# 初始化 Session State
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = []
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = pd.DataFrame()
if 'selected_symbol' not in st.session_state:
    st.session_state['selected_symbol'] = ""

@st.cache_data(ttl=86400)
def get_taiwan_stock_list():
    """從證交所與櫃買中心抓取對照表 (具備雲端防錯機制)"""
    try:
        def process_df(url, suffix):
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, verify=False, timeout=10)
            res.encoding = 'big5' 
            dfs = pd.read_html(io.StringIO(res.text))
            df = dfs[0]
            col = df.columns[0]
            df = df[df[col].astype(str).str.contains('　')]
            split_data = df[col].astype(str).str.split('　', n=1, expand=True)
            split_data.columns = ['Ticker', 'Name']
            split_data['Symbol'] = split_data['Ticker'] + suffix
            split_data = split_data[split_data['Ticker'].str.isalnum()]
            return split_data

        df_tw = process_df("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", ".TW")
        df_two = process_df("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", ".TWO")
        return pd.concat([df_tw, df_two], ignore_index=True)
        
    except Exception:
        # 如果被證交所阻擋，不顯示紅框，改顯示溫和提示並回傳空表
        st.warning("ℹ️ 雲端主機目前無法連線至證交所取得「中文名稱對照表」。請直接在下方輸入代碼 (上市加 .TW，上櫃加 .TWO，如 2330.TW) 即可正常分析！")
        return pd.DataFrame()

def get_stock_data(ticker):
    """取得股票歷史資料並計算進階技術指標"""
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    info = stock.info
    
    if not hist.empty:
        hist['5MA'] = hist['Close'].rolling(window=5).mean()
        hist['20MA'] = hist['Close'].rolling(window=20).mean()
        hist['9MA_Max'] = hist['High'].rolling(window=9).max()
        hist['9MA_Min'] = hist['Low'].rolling(window=9).min()
        hist['RSV'] = 100 * (hist['Close'] - hist['9MA_Min']) / (hist['9MA_Max'] - hist['9MA_Min'])
        hist['K'] = hist['RSV'].ewm(com=2, adjust=False).mean()
        hist['D'] = hist['K'].ewm(com=2, adjust=False).mean()
        hist['EMA12'] = hist['Close'].ewm(span=12, adjust=False).mean()
        hist['EMA26'] = hist['Close'].ewm(span=26, adjust=False).mean()
        hist['DIF'] = hist['EMA12'] - hist['EMA26']
        hist['MACD'] = hist['DIF'].ewm(span=9, adjust=False).mean()
        hist['OSC'] = hist['DIF'] - hist['MACD']
        
    return hist.tail(120), info

def plot_kline(hist, ticker_name):
    """繪製包含 K線、MA、MACD、KD 的多重圖表"""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, 
                        row_heights=[0.5, 0.25, 0.25])
    
    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['5MA'], line=dict(color='orange', width=1.5), name='5MA'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['20MA'], line=dict(color='blue', width=1.5), name='20MA'), row=1, col=1)
    
    colors = ['red' if val >= 0 else 'green' for val in hist['OSC']]
    fig.add_trace(go.Bar(x=hist.index, y=hist['OSC'], marker_color=colors, name='MACD柱狀圖'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['DIF'], line=dict(color='orange', width=1.5), name='DIF'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=hist.index, y=hist['K'], line=dict(color='orange', width=1.5), name='K值'), row=3, col=1)
    fig.add_trace(go.Scatter(x=hist.index, y=hist['D'], line=dict(color='blue', width=1.5), name='D值'), row=3, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="green", row=3, col=1)

    fig.update_layout(title=f"{ticker_name} 進階技術分析", xaxis_rangeslider_visible=False, height=750)
    return fig

def generate_gemini_analysis(ticker, hist, info, api_key):
    """呼叫 Gemini 產生分析報告 (已切換為最穩定的 gemini-pro 模型)"""
    try:
        genai.configure(api_key=api_key)
        # 🔑 重點修改：強制雲端環境使用全球通用、最穩定的模型
        model = genai.GenerativeModel('gemini-pro') 
        
        latest = hist.iloc[-1]
        prompt = f"""
        你是一位極具實戰經驗的台股操盤手。請根據以下 {ticker} 的最新進階技術與基本面數據，撰寫一份約 400 字的實戰分析報告。
        【最新關鍵數據】
        - 收盤價：{latest['Close']:.2f}
        - 均線狀態：5日線({latest['5MA']:.2f}), 月線({latest['20MA']:.2f})
        - KD指標：K值({latest['K']:.2f}), D值({latest['D']:.2f})
        - MACD指標：DIF({latest['DIF']:.2f}), MACD({latest['MACD']:.2f}), 柱狀圖OSC({latest['OSC']:.2f})
        - 評價面：PB比({info.get('priceToBook', 'N/A')}), PE比({info.get('trailingPE', 'N/A')})
        
        【報告要求】：
        1. 📈 多指標綜合診斷：趨勢與動能評估。
        2. 🎯 進場時機與條件：尚未持股者的建倉建議與條件。
        3. 🛑 離場時機：已持股者的停損/停利防守點及技術訊號。
        4. 💡 總結建議：一句話總結策略。
        """
        response = model.generate_content(prompt)
        return response.text, latest
    except Exception as e:
        return f"⚠️ AI 分析錯誤：{e}", None

# --- 載入台股清單 ---
stock_df = get_taiwan_stock_list()

# --- 網頁側邊欄 ---
with st.sidebar:
    st.header("🔑 設定區")
    api_key = st.text_input("輸入 Gemini API Key", type="password")
    
    st.markdown("---")
    st.header("⭐ 我的最愛")
    if st.session_state['favorites']:
        for fav_dict in st.session_state['favorites']:
            display_name = f"{fav_dict['Ticker']} {fav_dict['Name']}"
            if st.button(display_name, key=f"btn_{fav_dict['Ticker']}"):
                st.session_state['selected_symbol'] = fav_dict['Symbol']
                st.session_state['display_name'] = display_name
    else:
        st.write("目前尚無最愛名單。")

# --- 網頁主畫面 ---
st.title("📊 股票進階指標與 AI 分析系統")
st.markdown("支援輸入 **股票名稱** (如: 友達) 或 **代碼** (如: 2409.TW)")

# 搜尋區塊
col_s1, col_s2 = st.columns([4, 1])
with col_s1:
    search_query = st.text_input("輸入名稱或代碼 (若查無選單請直接輸入代碼如 2330.TW)：", key="search_input")
with col_s2:
    st.write("") 
    st.write("")
    if st.button("🔍 搜尋標的"):
        # 如果有載入清單，則進行模糊搜尋
        if not stock_df.empty and search_query:
            mask = stock_df['Ticker'].str.contains(search_query) | stock_df['Name'].str.contains(search_query)
            st.session_state['search_results'] = stock_df[mask]
        else:
            # 若無清單 (被阻擋)，直接將輸入的內容作為代碼去分析
            st.session_state['selected_symbol'] = search_query.upper()
            st.session_state['display_name'] = search_query.upper()

# 選單區塊 (有抓到對照表才顯示)
if not st.session_state['search_results'].empty:
    st.markdown("---")
    results_df = st.session_state['search_results']
    options = results_df['Ticker'] + " " + results_df['Name'] + " (" + results_df['Symbol'] + ")"
    
    selected_option = st.selectbox("👉 請選擇您要分析的精確標的：", options.tolist())
    
    col_a1, col_a2 = st.columns([1, 5])
    with col_a1:
        analyze_btn = st.button("進行深度分析", type="primary")
    with col_a2:
        if st.button("加入我的最愛 ❤️"):
            sel_ticker = selected_option.split(" ")[0]
            sel_name = selected_option.split(" ")[1]
            sel_symbol = selected_option.split("(")[-1].replace(")", "")
            
            fav_item = {"Ticker": sel_ticker, "Name": sel_name, "Symbol": sel_symbol}
            if fav_item not in st.session_state['favorites']:
                st.session_state['favorites'].append(fav_item)
                st.success(f"已將 {sel_name} 加入最愛！")
                st.rerun()
                
    if analyze_btn:
        st.session_state['selected_symbol'] = selected_option.split("(")[-1].replace(")", "")
        st.session_state['display_name'] = selected_option.split(" (")[0]
elif search_query and stock_df.empty:
    # 沒抓到表單時的直接分析按鈕
    if st.button("進行深度分析", type="primary"):
        st.session_state['selected_symbol'] = search_query.upper()
        st.session_state['display_name'] = search_query.upper()

# 分析與圖表區塊
if st.session_state['selected_symbol']:
    st.markdown("---")
    target_symbol = st.session_state['selected_symbol']
    display_title = st.session_state.get('display_name', target_symbol)
    
    with st.spinner(f'正在分析 {display_title} 的進階指標與產出報告...'):
        hist_data, stock_info = get_stock_data(target_symbol)
        
        if not hist_data.empty:
            st.plotly_chart(plot_kline(hist_data, display_title), use_container_width=True)
            
            st.markdown("### 🤖 專屬 AI 操盤手報告")
            if api_key:
                ai_report, latest_data = generate_gemini_analysis(display_title, hist_data, stock_info, api_key)
                st.info(ai_report)
                
                if latest_data is not None:
                    export_content = f"【{display_title} AI 實戰分析】\n產出時間:{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{ai_report}"
                    st.download_button("📥 下載分析報告 (TXT)", data=export_content, file_name=f"{target_symbol}_AI_Report.txt")
            else:
                st.warning("請在側邊欄輸入 Gemini API Key 啟用 AI 分析。")
                
            with st.expander("查看近期數據"):
                display_df = hist_data[['Close', 'Volume', '5MA', '20MA', 'K', 'D', 'MACD', 'OSC']].tail(10).sort_index(ascending=False)
                st.dataframe(display_df.round(2))
        else:
            st.error("擷取股價資料失敗，請確認輸入的代碼格式 (如: 2330.TW)。")
