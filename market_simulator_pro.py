"""
市场波动监测与预测模拟器（终极简化版）
功能：
- AI新闻分析（OpenAI，可降级模拟）
- 纯Python技术指标（RSI、MACD、布林带）
- 回撤与风险指标
- 因子分析（Alpha/Beta）
- 策略回测（双均线示例）
- 线性回归价格预测（可扩展为LSTM，但需额外安装）
- Streamlit Web仪表盘

特点：依赖极少，部署简单，兼容Python 3.8-3.11
"""

import os
import json
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ---------------------------- 第三方库导入（带降级）---------------------------
try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("⚠ streamlit/plotly未安装，无法启动Web界面")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ openai库未安装，AI分析将使用模拟模式")

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠ scikit-learn未安装，预测功能将降级")

# ---------------------------- 配置参数 ----------------------------
INITIAL_PRICE = 100.0
DAYS = 30
VOLATILITY = 0.02
NEWS_IMPACT = 0.01
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# OpenAI API密钥（从环境变量或Streamlit Secrets读取）
if STREAMLIT_AVAILABLE:
    # 如果运行在Streamlit，尝试从secrets获取
    try:
        OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except:
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai.api_key = OPENAI_API_KEY
    AI_ACTIVE = True
else:
    AI_ACTIVE = False
    print("⚠ OpenAI未配置，使用模拟新闻")

# ---------------------------- 纯Python技术指标计算 ----------------------------
def compute_rsi(prices, period=14):
    """计算RSI"""
    if len(prices) < period + 1:
        return 50
    deltas = np.diff(prices[-period-1:])
    seed = deltas[:period]
    up = seed[seed > 0].sum() / period
    down = -seed[seed < 0].sum() / period
    if down == 0:
        return 100
    rs = up / down
    return 100 - 100 / (1 + rs)

def compute_macd(prices, fast=12, slow=26, signal=9):
    """简化MACD（使用EMA近似）"""
    if len(prices) < slow + signal:
        return 0, 0, 0
    def ema(data, window):
        alpha = 2 / (window + 1)
        ema_values = [data[0]]
        for price in data[1:]:
            ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
        return np.array(ema_values)

    close = np.array(prices)
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line[-1], signal_line[-1], histogram[-1]

def compute_bollinger_bands(prices, period=20, num_std=2):
    """布林带"""
    if len(prices) < period:
        return prices[-1], prices[-1], prices[-1], 0
    recent = prices[-period:]
    mean = np.mean(recent)
    std = np.std(recent)
    upper = mean + num_std * std
    lower = mean - num_std * std
    width = (upper - lower) / mean
    return upper, mean, lower, width

# ---------------------------- 模拟器核心类 ----------------------------
class MarketSimulator:
    def __init__(self, stock_name="演示股票"):
        self.stock_name = stock_name
        self.current_price = INITIAL_PRICE
        self.historical_prices = [INITIAL_PRICE]
        start_date = datetime.now().date() - timedelta(days=DAYS)
        self.historical_dates = [start_date]
        self.news_log = []
        self.predictions = []
        self.technical_indicators = {}
        self.factors = {}
        self.log("=== 模拟器初始化 ===")

    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)
        log_file = os.path.join(LOG_DIR, f"market_log_{datetime.now():%Y%m%d}.txt")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    # -------------------- AI新闻（真实或模拟）--------------------
    def fetch_ai_news(self):
        """获取新闻并使用AI分析（若可用）"""
        news_list = [
            "央行宣布降息25个基点，释放流动性",
            "科技巨头发布超预期财报，营收增长30%",
            "监管部门出台数据安全新规，加强行业规范",
            "国际油价大幅上涨，通胀担忧加剧",
            "地缘政治紧张局势升级，避险情绪升温",
            "新能源补贴政策延续，行业迎利好",
            "消费信心指数回升，零售数据超预期"
        ]
        headline = random.choice(news_list)

        if AI_ACTIVE:
            try:
                prompt = f"请分析以下新闻对股市的影响，返回JSON格式：新闻：{headline}，字段：sentiment（-1到1的浮点数），analysis（简短中文分析），impact（正面/负面/中性）"
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=200
                )
                result = json.loads(response.choices[0].message.content)
                sentiment = result.get("sentiment", 0)
                analysis = result.get("analysis", "")
                impact = result.get("impact", "中性")
            except Exception as e:
                self.log(f"AI调用失败: {e}", level="ERROR")
                sentiment = random.uniform(-0.5, 0.5)
                analysis = "AI服务暂时不可用，使用随机情绪"
                impact = "正面" if sentiment > 0.2 else "负面" if sentiment < -0.2 else "中性"
        else:
            sentiment = random.uniform(-0.5, 0.5)
            analysis = "模拟AI分析（未配置真实API）"
            impact = "正面" if sentiment > 0.2 else "负面" if sentiment < -0.2 else "中性"

        log_msg = f"新闻: {headline} | 分析: {analysis} | 情绪: {impact} (得分: {sentiment:.2f})"
        self.news_log.append(log_msg)
        self.log(f"AI新闻: {log_msg}")
        return sentiment

    # -------------------- 市场更新 --------------------
    def update_market(self):
        news_sentiment = self.fetch_ai_news()
        random_shock = np.random.normal(0, VOLATILITY)
        total_return = random_shock + news_sentiment * NEWS_IMPACT
        self.current_price *= (1 + total_return)
        self.current_price = max(self.current_price, 0.01)

        self.historical_prices.append(self.current_price)
        self.historical_dates.append(self.historical_dates[-1] + timedelta(days=1))

        if len(self.historical_prices) >= 5:
            recent_returns = np.diff(self.historical_prices[-5:]) / self.historical_prices[-5:-1]
            vol = np.std(recent_returns)
            self.log(f"价格: {self.current_price:.2f}, 近5日波动率: {vol:.4f}")
            if vol > 0.05:
                self.log("⚠ 市场波动过大！", level="WARNING")

    # -------------------- 技术指标更新 --------------------
    def update_indicators(self):
        prices = self.historical_prices
        ind = {}
        ind['RSI'] = compute_rsi(prices)
        macd, signal, hist = compute_macd(prices)
        ind['MACD'] = macd
        ind['MACD_signal'] = signal
        ind['MACD_hist'] = hist
        bb_up, bb_mid, bb_low, bb_width = compute_bollinger_bands(prices)
        ind['BB_upper'] = bb_up
        ind['BB_middle'] = bb_mid
        ind['BB_lower'] = bb_low
        ind['BB_width'] = bb_width
        self.technical_indicators = ind
        self.log(f"技术指标: RSI={ind['RSI']:.1f}, 布林带宽={ind['BB_width']:.3f}")

    # -------------------- 因子分析 --------------------
    def compute_factors(self):
        prices = np.array(self.historical_prices)
        if len(prices) < 2:
            return
        returns = np.diff(prices) / prices[:-1]
        # 模拟基准收益率（沪深300模拟）
        np.random.seed(42)
        benchmark = np.random.normal(0.0005, 0.01, len(returns))
        # 线性回归计算Beta
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(benchmark.reshape(-1,1), returns)
        beta = model.coef_[0]
        alpha = np.mean(returns) - beta * np.mean(benchmark)
        alpha_annual = alpha * 252
        self.factors = {
            'alpha': alpha,
            'alpha_annual': alpha_annual,
            'beta': beta,
            'annual_return': (prices[-1]/prices[0]) ** (252/len(prices)) - 1
        }
        self.log(f"因子: Alpha(年化)={alpha_annual:.4f}, Beta={beta:.2f}")

    # -------------------- 回撤与风险 --------------------
    def calculate_drawdown(self):
        prices = np.array(self.historical_prices)
        if len(prices) < 2:
            return {}
        returns = np.diff(prices) / prices[:-1]
        cumulative = np.cumprod(1 + np.concatenate([[0], returns]))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_dd = np.max(drawdown)
        # 夏普比率
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        # 索提诺比率
        downside = returns[returns < 0]
        sortino = np.mean(returns) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) != 0 else 0
        metrics = {
            'max_drawdown': max_dd,
            'max_drawdown_pct': f"{max_dd*100:.2f}%",
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'annual_volatility': np.std(returns) * np.sqrt(252)
        }
        if max_dd > 0.1:
            self.log(f"⚠ 最大回撤 {max_dd*100:.1f}%", level="重点标记")
        return metrics

    # -------------------- 线性回归预测 --------------------
    def predict_next_price(self):
        if len(self.historical_prices) < 5:
            return None
        # 用最近5天拟合线性趋势
        x = np.arange(len(self.historical_prices[-5:])).reshape(-1,1)
        y = self.historical_prices[-5:]
        model = LinearRegression()
        model.fit(x, y)
        next_idx = np.array([[5]])
        pred = model.predict(next_idx)[0]
        self.log(f"线性回归预测下一日价格: {pred:.2f}")
        return pred

    # -------------------- 策略回测（双均线示例）--------------------
    def backtest_ma_strategy(self, short=5, long=20):
        prices = np.array(self.historical_prices)
        if len(prices) < long:
            return {}
        # 简单移动平均
        ma_short = np.convolve(prices, np.ones(short)/short, mode='valid')
        ma_long = np.convolve(prices, np.ones(long)/long, mode='valid')
        min_len = min(len(ma_short), len(ma_long))
        ma_short = ma_short[-min_len:]
        ma_long = ma_long[-min_len:]
        # 生成信号：金叉买入(1)，死叉卖出(-1)
        signals = np.zeros(min_len)
        for i in range(1, min_len):
            if ma_short[i-1] <= ma_long[i-1] and ma_short[i] > ma_long[i]:
                signals[i] = 1
            elif ma_short[i-1] >= ma_long[i-1] and ma_short[i] < ma_long[i]:
                signals[i] = -1
        # 对齐价格
        aligned_prices = prices[-min_len:]
        # 模拟交易
        position = 0
        capital = 10000
        trades = []
        for i in range(min_len):
            if signals[i] == 1 and position == 0:
                position = capital / aligned_prices[i]
                capital = 0
                trades.append(('buy', aligned_prices[i]))
            elif signals[i] == -1 and position > 0:
                capital = position * aligned_prices[i]
                position = 0
                trades.append(('sell', aligned_prices[i]))
        final_value = capital + position * aligned_prices[-1]
        total_return = (final_value - 10000) / 10000
        # 胜率
        if len(trades) >= 2:
            buy_prices = [t[1] for t in trades if t[0]=='buy']
            sell_prices = [t[1] for t in trades if t[0]=='sell']
            wins = sum(s > b for s,b in zip(sell_prices, buy_prices))
            win_rate = wins / len(buy_prices) if buy_prices else 0
        else:
            win_rate = 0
        result = {
            '总收益': f"{total_return*100:.2f}%",
            '交易次数': len(trades)//2,
            '胜率': f"{win_rate*100:.2f}%",
            '最终资产': f"{final_value:.2f}"
        }
        self.log(f"回测结果: {result}")
        return result

    # -------------------- 运行模拟 --------------------
    def run_simulation(self, days=DAYS):
        self.log("=== 模拟开始 ===")
        for day in range(1, days+1):
            self.log(f"--- 第{day}天 ---")
            self.update_market()
            if day % 5 == 0:
                self.update_indicators()
                self.compute_factors()
                metrics = self.calculate_drawdown()
                self.log(f"风险: {metrics}")
        # 最终预测
        self.predict_next_price()
        self.log("=== 模拟结束 ===")


# ---------------------------- Streamlit Web界面 ----------------------------
def main():
    st.set_page_config(layout="wide", page_title="市场波动模拟器")
    st.title("📈 市场波动监测与预测模拟器（极简版）")

    # 初始化或加载模拟器
    if 'sim' not in st.session_state:
        st.session_state.sim = MarketSimulator()
        with st.spinner("正在运行30天模拟..."):
            st.session_state.sim.run_simulation(days=30)
    sim = st.session_state.sim

    # 侧边栏控制
    st.sidebar.header("控制面板")
    if st.sidebar.button("重新运行模拟 (30天)"):
        sim = MarketSimulator()
        sim.run_simulation(days=30)
        st.session_state.sim = sim
        st.success("模拟完成！")

    st.sidebar.subheader("AI配置")
    api_key = st.sidebar.text_input("OpenAI API密钥（可选）", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.sidebar.success("密钥已设置，下次新闻将使用真实AI")

    # 主标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📊 市场概览", "📈 技术指标", "📉 风险与回撤", "⚙️ 回测与预测"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("当前价格", f"${sim.current_price:.2f}",
                     delta=f"{(sim.current_price/sim.historical_prices[-2]-1)*100:.2f}%" if len(sim.historical_prices)>1 else None)
            st.metric("历史天数", len(sim.historical_prices))
        with col2:
            if len(sim.historical_prices) > 1:
                returns = np.diff(sim.historical_prices) / sim.historical_prices[:-1]
                st.metric("年化波动率", f"{np.std(returns)*np.sqrt(252)*100:.2f}%")
        # 价格走势图
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sim.historical_dates, y=sim.historical_prices, mode='lines+markers', name='价格'))
        fig.update_layout(title='历史价格走势', xaxis_title='日期', yaxis_title='价格')
        st.plotly_chart(fig, use_container_width=True)
        # 最近新闻
        st.subheader("最近AI新闻分析")
        for news in sim.news_log[-5:]:
            st.info(news)

    with tab2:
        if sim.technical_indicators:
            ind = sim.technical_indicators
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("RSI(14)", f"{ind['RSI']:.1f}")
                st.metric("MACD", f"{ind['MACD']:.2f}")
            with col2:
                st.metric("布林带上轨", f"{ind['BB_upper']:.2f}")
                st.metric("布林带中轨", f"{ind['BB_middle']:.2f}")
            with col3:
                st.metric("布林带下轨", f"{ind['BB_lower']:.2f}")
                st.metric("布林带宽", f"{ind['BB_width']:.3f}")
        else:
            st.write("暂无技术指标数据（需至少20天数据）")

    with tab3:
        metrics = sim.calculate_drawdown()
        if metrics:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最大回撤", metrics['max_drawdown_pct'])
                st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
            with col2:
                st.metric("索提诺比率", f"{metrics['sortino_ratio']:.2f}")
                st.metric("年化波动", f"{metrics['annual_volatility']*100:.2f}%")
            # 回撤曲线
            prices = np.array(sim.historical_prices)
            returns = np.diff(prices) / prices[:-1]
            cumulative = np.cumprod(1 + np.concatenate([[0], returns]))
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sim.historical_dates, y=drawdown*100, fill='tozeroy', name='回撤%'))
            fig.update_layout(title='回撤曲线', yaxis_title='回撤 (%)')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("策略回测（双均线）")
        if st.button("运行双均线回测"):
            result = sim.backtest_ma_strategy()
            st.json(result)
        st.subheader("价格预测")
        if st.button("线性回归预测下一日"):
            pred = sim.predict_next_price()
            if pred:
                st.metric("预测价格", f"${pred:.2f}")
            else:
                st.warning("数据不足")
        # 显示最新预测
        if sim.predictions:
            st.subheader("历史预测")
            st.write(sim.predictions[-1])


# ---------------------------- 入口 ----------------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        main()
    else:
        # 命令行模式
        sim = MarketSimulator()
        sim.run_simulation(days=30)
        print("\n模拟完成，日志保存在logs文件夹。")
        print("要启动Web仪表盘，请安装streamlit和plotly：pip install streamlit plotly")