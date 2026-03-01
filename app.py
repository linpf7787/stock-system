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
import sqlite3
import hashlib

# 隱藏略過 SSL 驗證時產生的警告訊息
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 設定網頁標題與佈局
st.set_page_config(page_title="AI 股票分析與進階指標系統 (多用戶版)", layout="wide")

# ==========================================
# 🗄️ 資料庫設定與操作函數
# ==========================================
def init_db():
    """初始化 SQLite 資料庫與資料表"""
    conn = sqlite3.connect('stock_app.db', check_same_thread=False)
    c = conn.cursor()
    # 用戶表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT UNIQUE, 
                  password TEXT, 
                  api_key TEXT)''')
    # 我的最愛表 (紀錄報告與時間)
    c.execute('''CREATE TABLE IF NOT EXISTS favorites
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  user_id INTEGER, 
                  ticker TEXT, 
                  name TEXT, 
                  symbol TEXT, 
                  report TEXT, 
                  update_time TEXT)''')
    conn.commit()
    conn.close()

def hash_pw(password):
    """將密碼進行 SHA-256 加密"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    conn = sqlite3.connect('stock_app.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password, api_key) VALUES (?, ?, '')", (username, hash_pw(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False 
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('stock_app.db')
    c = conn.cursor()
    c.execute("SELECT id, api_key FROM users WHERE username=? AND password=?", (username, hash_pw(password)))
    user = c.fetchone()
    conn.close()
    return user 

def update_api_key_db(user_id, api_key):
    conn = sqlite3.connect('stock_app.db')
    c = conn.cursor()
    c.execute("UPDATE users SET api_key=? WHERE id=?", (api_key, user_id))
    conn.commit()
    conn.close()

def get_favorites(user_id):
    conn = sqlite3.connect('stock_app.db')
    df = pd.read_sql_query("SELECT * FROM favorites WHERE user_id=?", conn, params=(user_id,))
    conn.close()
    return df

def save_favorite_report(user_id, ticker, name, symbol, report):
    """新增或更新我的最愛中的分析報告"""
    conn = sqlite3.connect('stock_app.db')
    c = conn.cursor()
    update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    c.execute("SELECT id FROM favorites WHERE user_id=? AND symbol=?", (user_id, symbol))
    row = c.fetchone()
    if row:
        c.execute("UPDATE favorites SET report=?, update_time=? WHERE id=?", (report, update_time, row[0]))
    else:
        c.execute("INSERT INTO favorites (user_id, ticker, name, symbol, report, update_time) VALUES (?, ?, ?, ?, ?, ?)",
                  (user_id, ticker, name, symbol, report, update_time))
    conn.commit()
    conn.close()

def remove_favorite(user_id, symbol):
    conn = sqlite3.connect('stock_app.db')
    c = conn.cursor()
    c.execute("DELETE FROM favorites WHERE user_id=? AND symbol=?", (user_id, symbol))
    conn.commit()
    conn.close()

init_db()

# ==========================================
# 🔄 股票資料與 AI 分析函數
# ==========================================
@st.cache_data(ttl=86400)
def get_taiwan_stock_list():
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
        return pd.DataFrame()

def get_stock_data(ticker):
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
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25])
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
    try:
        genai.configure(api_key=api_key)
        valid_model_name = ""
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                if 'flash' in m.name or 'pro' in m.name:
                    valid_model_name = m.name
                    break
        if not valid_model_name:
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    valid_model_name = m.name
                    break
        if not valid_model_name:
            return "⚠️ 您的 API Key 無法存取文字生成模型。", None

        model = genai.GenerativeModel(valid_model_name) 
        latest = hist.iloc[-1]
        prompt = f"""
        你是一位極具實戰經驗的台股操盤手。請根據以下 {ticker} 的最新進階技術與基本面數據，撰寫約 400 字的實戰分析報告。
        - 收盤價：{latest['Close']:.2f} | 5MA({latest['5MA']:.2f}) | 20MA({latest['20MA']:.2f})
        - KD指標：K({latest['K']:.2f}) | D({latest['D']:.2f})
        - MACD：DIF({latest['DIF']:.2f}) | MACD({latest['MACD']:.2f}) | OSC({latest['OSC']:.2f})
        - 評價面：PB比({info.get('priceToBook', 'N/A')}), PE比({info.get('trailingPE', 'N/A')})
        要求包含：1.多指標綜合診斷 2.進場時機與條件 3.離場/防守點時機 4.一句話總結。
        """
        response = model.generate_content(prompt)
        return f"*(模型: {valid_model_name})*\n\n" + response.text, latest
    except Exception as e:
        return f"⚠️ AI 分析錯誤：{e}", None

# ==========================================
# 🖥️ 使用者介面與流程控制
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ""

if not st.session_state['logged_in']:
    st.title("🔐 AI 股票分析系統 - 登入")
    tab1, tab2 = st.tabs(["登入", "註冊新帳號"])
    
    with tab1:
        login_user = st.text_input("帳號", key="login_user")
        login_pw = st.text_input("密碼", type="password", key="login_pw")
        if st.button("登入系統", type="primary"):
            user = authenticate_user(login_user, login_pw)
            if user:
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = user[0]
                st.session_state['username'] = login_user
                st.session_state['api_key'] = user[1] if user[1] else ""
                st.success("登入成功！")
                st.rerun()
            else:
                st.error("帳號或密碼錯誤。")
                
    with tab2:
        reg_user = st.text_input("設定帳號", key="reg_user")
        reg_pw = st.text_input("設定密碼", type="password", key="reg_pw")
        reg_pw2 = st.text_input("確認密碼", type="password", key="reg_pw2")
        if st.button("註冊"):
            if reg_pw != reg_pw2:
                st.error("兩次密碼輸入不一致")
            elif reg_user and reg_pw:
                if register_user(reg_user, reg_pw):
                    st.success("註冊成功！請切換到「登入」標籤登入系統。")
                else:
                    st.error("此帳號已被使用，請換一個。")
    st.stop() 

if 'search_results' not in st.session_state:
    st.session_state['search_results'] = pd.DataFrame()
if 'selected_symbol' not in st.session_state:
    st.session_state['selected_symbol'] = ""
if 'current_report' not in st.session_state:
    st.session_state['current_report'] = ""
if 'report_time' not in st.session_state:
    st.session_state['report_time'] = ""
if 'needs_new_analysis' not in st.session_state:
    st.session_state['needs_new_analysis'] = False

stock_df = get_taiwan_stock_list()

# --- 側邊欄 ---
with st.sidebar:
    st.markdown(f"### 👤 歡迎, **{st.session_state['username']}**")
    if st.button("🚪 登出系統"):
        st.session_state.clear()
        st.rerun()
        
    st.markdown("---")
    st.header("🔑 API 設定")
    new_api_key = st.text_input("Gemini API Key", value=st.session_state['api_key'], type="password")
    if new_api_key != st.session_state['api_key']:
        st.session_state['api_key'] = new_api_key
        update_api_key_db(st.session_state['user_id'], new_api_key)
        st.success("API Key 已儲存！")
    
    st.markdown("---")
    st.header("⭐ 我的最愛")
    fav_df = get_favorites(st.session_state['user_id'])
    
    if not fav_df.empty:
        for index, row in fav_df.iterrows():
            # 🐛 這裡已修正：全改用小寫欄位名稱 (ticker, name, symbol, report)
            if st.button(f"{row['ticker']} {row['name']}", key=f"fav_{row['symbol']}", use_container_width=True):
                st.session_state['selected_symbol'] = row['symbol']
                st.session_state['display_name'] = f"{row['ticker']} {row['name']}"
                st.session_state['current_report'] = row['report']
                st.session_state['report_time'] = row['update_time']
                st.session_state['needs_new_analysis'] = False
    else:
        st.write("尚無收藏。在分析後可點擊愛心加入。")

# --- 主畫面區塊 ---
st.title("📊 股票進階指標與 AI 分析系統")

col_s1, col_s2 = st.columns([4, 1])
with col_s1:
    search_query = st.text_input("輸入名稱或代碼 (如: 友達, 2409.TW)：", key="search_input")
with col_s2:
    st.write("") 
    st.write("")
    if st.button("🔍 搜尋標的"):
        if not stock_df.empty and search_query:
            mask = stock_df['Ticker'].str.contains(search_query) | stock_df['Name'].str.contains(search_query)
            st.session_state['search_results'] = stock_df[mask]
        else:
            st.session_state['selected_symbol'] = search_query.upper()
            st.session_state['display_name'] = search_query.upper()
            st.session_state['needs_new_analysis'] = True

if not st.session_state['search_results'].empty:
    st.markdown("---")
    results_df = st.session_state['search_results']
    options = results_df['Ticker'] + " " + results_df['Name'] + " (" + results_df['Symbol'] + ")"
    selected_option = st.selectbox("👉 請選擇精確標的：", options.tolist())
    
    if st.button("進行深度分析", type="primary"):
        st.session_state['selected_symbol'] = selected_option.split("(")[-1].replace(")", "")
        st.session_state['display_name'] = selected_option.split(" (")[0]
        st.session_state['needs_new_analysis'] = True

if st.session_state['selected_symbol']:
    st.markdown("---")
    target_symbol = st.session_state['selected_symbol']
    display_title = st.session_state.get('display_name', target_symbol)
    user_id = st.session_state['user_id']
    
    hist_data, stock_info = get_stock_data(target_symbol)
    
    if not hist_data.empty:
        st.plotly_chart(plot_kline(hist_data, display_title), use_container_width=True)
        st.markdown("### 🤖 專屬 AI 操盤手報告")
        
        if st.session_state['needs_new_analysis']:
            if st.session_state['api_key']:
                with st.spinner('正在為您產生最新 AI 分析...'):
                    ai_report, latest_data = generate_gemini_analysis(display_title, hist_data, stock_info, st.session_state['api_key'])
                    st.session_state['current_report'] = ai_report
                    st.session_state['report_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    st.session_state['needs_new_analysis'] = False
                    
                    # 🐛 這裡已修正：改用小寫的 symbol
                    if target_symbol in fav_df['symbol'].values:
                        ticker = display_title.split(" ")[0] if " " in display_title else target_symbol
                        save_favorite_report(user_id, ticker, display_title, target_symbol, ai_report)
            else:
                st.warning("⚠️ 請先在左側邊欄設定您的 Gemini API Key。")
        
        if st.session_state['current_report']:
            st.caption(f"🕒 報告產出時間: {st.session_state['report_time']} (圖表報價為即時讀取)")
            st.info(st.session_state['current_report'])
            
            # 🐛 這裡已修正：改用小寫的 symbol
            is_fav = target_symbol in fav_df['symbol'].values
            
            col_b1, col_b2, col_b3 = st.columns([2, 2, 4])
            with col_b1:
                if is_fav:
                    if st.button("🔄 重新取得最新 AI 分析", use_container_width=True):
                        st.session_state['needs_new_analysis'] = True
                        st.rerun()
            with col_b2:
                if is_fav:
                    if st.button("💔 移除最愛", use_container_width=True):
                        remove_favorite(user_id, target_symbol)
                        st.success("已移除！")
                        st.rerun()
                else:
                    if st.button("❤️ 將此報告加入我的最愛", use_container_width=True, type="primary"):
                        ticker = display_title.split(" ")[0] if " " in display_title else target_symbol
                        save_favorite_report(user_id, ticker, display_title, target_symbol, st.session_state['current_report'])
                        st.success("已儲存報告至最愛！")
                        st.rerun()
            
            export_content = f"【{display_title} AI 實戰分析】\n產出時間:{st.session_state['report_time']}\n\n{st.session_state['current_report']}"
            st.download_button("📥 下載此份報告 (TXT)", data=export_content, file_name=f"{target_symbol}_AI_Report.txt")

        with st.expander("查看近期數據明細"):
            display_df = hist_data[['Close', 'Volume', '5MA', '20MA', 'K', 'D', 'MACD', 'OSC']].tail(10).sort_index(ascending=False)
            st.dataframe(display_df.round(2))
    else:
        st.error("擷取股價資料失敗，請確認代碼是否正確。")
