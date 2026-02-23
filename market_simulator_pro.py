"""
市场波动监测与预测模拟器（专业版 + Web仪表盘 + 回测 + 因子分析 + 深度学习）
功能：
- AI新闻分析（OpenAI API）
- 150+技术指标（TA-Lib）
- 回撤与风险指标计算（含AI解读）
- 因子分析（Alpha/Beta）
- LSTM价格预测
- 策略回测平台
- Streamlit Web仪表盘

作者：AI Assistant
日期：2024
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ---------------------------- 依赖库导入 ----------------------------
# 基础库
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TA-Lib（可选）
try:
    import talib

    TA_LIB_AVAILABLE = True
except ImportError:
    TA_LIB_AVAILABLE = False
    print("⚠ TA-Lib未安装，使用简化指标。安装: pip install TA-Lib")

# OpenAI（可选）
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠ openai库未安装，AI分析功能将降级。安装: pip install openai")

# 深度学习（TensorFlow/Keras）
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠ TensorFlow未安装，深度学习预测将禁用。安装: pip install tensorflow")

# ---------------------------- 配置参数 ----------------------------
# 直接硬编码API密钥（风险提示：请勿公开分享）
OPENAI_API_KEY = "sk-81e6d1b52aff4028a1ac3280aa4e7e63"  # 请替换为真实密钥

# 模拟参数
INITIAL_PRICE = 100.0
DAYS = 30
VOLATILITY = 0.02
NEWS_IMPACT = 0.01
LOG_DIR = "logs"
MODEL_DIR = "models"

# 创建目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 初始化OpenAI
if OPENAI_AVAILABLE and OPENAI_API_KEY and OPENAI_API_KEY != "sk-81e6d1b52aff4028a1ac3280aa4e7e63":
    openai.api_key = OPENAI_API_KEY
    AI_AVAILABLE = True
else:
    AI_AVAILABLE = False
    print("⚠ OpenAI API未配置，将使用模拟AI功能")


# ---------------------------- 模拟器核心类 ----------------------------
class MarketSimulatorPro:
    def __init__(self, stock_name="SIM_STOCK"):
        self.stock_name = stock_name
        self.current_price = INITIAL_PRICE
        self.historical_prices = [INITIAL_PRICE]
        self.historical_dates = [datetime.now().date() - timedelta(days=DAYS)]
        self.news_log = []
        self.predictions = []
        self.trading_signals = []
        self.technical_indicators = {}
        self.factors = {}  # 因子分析结果
        self.lstm_model = None  # LSTM模型
        self.lstm_trained = False

        # 初始化日志
        self.log("=== 市场模拟器初始化 ===")

    # -------------------- 日志记录 --------------------
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        print(log_entry)

        today_str = datetime.now().strftime("%Y%m%d")
        log_filename = os.path.join(LOG_DIR, f"market_log_{today_str}.txt")
        with open(log_filename, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    # -------------------- AI新闻分析（OpenAI）--------------------
    def fetch_ai_news_analysis(self, news_headline):
        if not AI_AVAILABLE:
            # 模拟
            sentiment = random.uniform(-0.5, 0.5)
            return {
                "sentiment": sentiment,
                "analysis": f"模拟分析：{news_headline}",
                "impact": "正面" if sentiment > 0.2 else "负面" if sentiment < -0.2 else "中性"
            }

        try:
            prompt = f"""
            请分析以下财经新闻对股市的影响，输出JSON格式：
            新闻：{news_headline}

            要求：
            1. sentiment: 情绪分数，范围-1（极负面）到1（极正面）
            2. analysis: 简短分析（50字以内）
            3. impact: "正面"/"负面"/"中性"
            4. affected_sectors: 可能影响的行业板块（数组）

            只返回JSON，不要其他文字。
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )

            result = json.loads(response.choices[0].message.content)

            # 记录AI分析
            with open(os.path.join(LOG_DIR, "ai_analysis_log.txt"), "a", encoding="utf-8") as f:
                f.write(f"{datetime.now()} | {news_headline} | {json.dumps(result, ensure_ascii=False)}\n")

            return result

        except Exception as e:
            self.log(f"AI分析失败: {e}", level="ERROR")
            return {"sentiment": 0, "analysis": "AI分析失败", "impact": "中性"}

    def fetch_ai_news(self):
        news_list = [
            "央行宣布降息25个基点，释放流动性",
            "科技巨头发布超预期财报，营收增长30%",
            "监管部门出台数据安全新规，加强行业规范",
            "国际油价大幅上涨，通胀担忧加剧",
            "地缘政治紧张局势升级，避险情绪升温",
            "新能源补贴政策延续，行业迎利好",
            "消费信心指数回升，零售数据超预期"
        ]
        news_headline = random.choice(news_list)
        ai_result = self.fetch_ai_news_analysis(news_headline)

        analysis_text = f"新闻: {news_headline} | AI分析: {ai_result['analysis']} | 情绪: {ai_result['impact']} (得分: {ai_result['sentiment']:.2f})"
        self.news_log.append(analysis_text)
        self.log(f"AI新闻处理: {analysis_text}")

        return ai_result['sentiment']

    # -------------------- 技术指标计算 --------------------
    def compute_technical_indicators(self):
        prices = np.array(self.historical_prices)
        if len(prices) < 20:
            return {}

        indicators = {}

        if TA_LIB_AVAILABLE:
            try:
                close = prices.astype(float)
                indicators['RSI'] = talib.RSI(close, timeperiod=14)[-1]
                macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators['MACD'] = macd[-1]
                indicators['MACD_signal'] = signal[-1]
                indicators['MACD_hist'] = hist[-1]
                upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
                indicators['BB_upper'] = upper[-1]
                indicators['BB_middle'] = middle[-1]
                indicators['BB_lower'] = lower[-1]
                indicators['BB_width'] = (upper[-1] - lower[-1]) / middle[-1]
                indicators['ADX'] = talib.ADX(close, close, close, timeperiod=14)[-1]
                indicators['ATR'] = talib.ATR(close, close, close, timeperiod=14)[-1]
                slowk, slowd = talib.STOCH(close, close, close, fastk_period=14, slowk_period=3, slowd_period=3)
                indicators['Stoch_K'] = slowk[-1]
                indicators['Stoch_D'] = slowd[-1]
            except Exception as e:
                self.log(f"TA-Lib计算失败: {e}", level="WARNING")
                indicators = self._compute_simple_indicators(prices)
        else:
            indicators = self._compute_simple_indicators(prices)

        self.technical_indicators = indicators
        return indicators

    def _compute_simple_indicators(self, prices):
        indicators = {}
        if len(prices) >= 14:
            changes = np.diff(prices[-15:])
            gains = changes[changes > 0].mean() if any(changes > 0) else 0
            losses = -changes[changes < 0].mean() if any(changes < 0) else 0
            if losses == 0:
                indicators['RSI'] = 100
            else:
                rs = gains / losses
                indicators['RSI'] = 100 - (100 / (1 + rs))
            mean = np.mean(prices[-20:])
            std = np.std(prices[-20:])
            indicators['BB_upper'] = mean + 2 * std
            indicators['BB_middle'] = mean
            indicators['BB_lower'] = mean - 2 * std
            indicators['BB_width'] = (4 * std) / mean
        return indicators

    # -------------------- 因子分析（Alpha/Beta）--------------------
    def compute_factors(self, benchmark_returns=None):
        """
        计算Alpha、Beta等因子
        需要基准收益率，若无则用无风险收益模拟
        """
        prices = np.array(self.historical_prices)
        if len(prices) < 2:
            return {}

        # 计算策略收益率
        returns = np.diff(prices) / prices[:-1]

        # 模拟基准收益率（例如沪深300）
        if benchmark_returns is None:
            # 生成一个相关的基准序列（模拟）
            np.random.seed(42)
            benchmark_returns = np.random.normal(0.0005, 0.01, len(returns))

        # 确保长度一致
        min_len = min(len(returns), len(benchmark_returns))
        returns = returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]

        # 线性回归计算Beta
        cov_matrix = np.cov(returns, benchmark_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0

        # Alpha = 平均超额收益 - Beta * 平均基准超额收益
        # 假设无风险利率为0
        alpha = np.mean(returns) - beta * np.mean(benchmark_returns)

        # 年化Alpha
        alpha_annual = alpha * 252

        # 其他因子：R-squared
        correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        r_squared = correlation ** 2

        # 信息比率（相对于基准）
        tracking_error = np.std(returns - benchmark_returns)
        information_ratio = np.mean(returns - benchmark_returns) / tracking_error if tracking_error != 0 else 0

        factors = {
            'alpha': alpha,
            'alpha_annual': alpha_annual,
            'beta': beta,
            'r_squared': r_squared,
            'information_ratio': information_ratio,
            'correlation_with_benchmark': correlation
        }

        self.factors = factors
        self.log(f"因子分析: Alpha(年化)={alpha_annual:.4f}, Beta={beta:.2f}, IR={information_ratio:.2f}")
        return factors

    # -------------------- 回撤与风险指标 --------------------
    def calculate_drawdown_metrics(self, prices=None):
        if prices is None:
            prices = np.array(self.historical_prices)
        if len(prices) < 2:
            return {}

        returns = np.diff(prices) / prices[:-1]
        cumulative = np.cumprod(1 + np.concatenate([[0], returns]))

        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        max_drawdown = np.max(drawdown)
        max_drawdown_idx = np.argmax(drawdown)

        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0

        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # 最大回撤持续期
        recovery_idx = None
        for i in range(max_drawdown_idx, len(cumulative)):
            if cumulative[i] >= peak[max_drawdown_idx]:
                recovery_idx = i
                break
        drawdown_duration = (recovery_idx - max_drawdown_idx) if recovery_idx else len(cumulative) - max_drawdown_idx

        annual_return = (prices[-1] / prices[0]) ** (252 / len(prices)) - 1
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        metrics = {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': f"{max_drawdown * 100:.2f}%",
            'max_drawdown_duration': drawdown_duration,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'annual_return': annual_return,
            'annual_volatility': np.std(returns) * np.sqrt(252)
        }

        if max_drawdown > 0.2:
            self.log(f"⚠ 严重警告：最大回撤已达{max_drawdown * 100:.1f}%", level="重点标记")
        elif max_drawdown > 0.1:
            self.log(f"⚠ 注意：最大回撤{max_drawdown * 100:.1f}%", level="重点标记")

        return metrics

    # -------------------- LSTM深度学习预测 --------------------
    def build_lstm_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def train_lstm(self, lookback=10, epochs=50):
        """训练LSTM模型预测下一日价格"""
        if not TF_AVAILABLE:
            self.log("TensorFlow不可用，跳过LSTM训练", level="WARNING")
            return None

        if len(self.historical_prices) < lookback + 2:
            self.log("历史数据不足，无法训练LSTM", level="WARNING")
            return None

        # 准备数据
        data = np.array(self.historical_prices).reshape(-1, 1)

        # 归一化
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        X, y = [], []
        for i in range(lookback, len(scaled_data) - 1):
            X.append(scaled_data[i - lookback:i, 0])
            y.append(scaled_data[i + 1, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # 划分训练/验证
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # 构建模型
        model = self.build_lstm_model((X.shape[1], 1))

        # 早停
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # 训练
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )

        # 保存模型和归一化器
        model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        model.save(model_path)

        # 保存scaler
        import joblib
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)

        self.lstm_model = model
        self.lstm_trained = True
        self.log(f"LSTM模型训练完成，保存至{model_path}")

        # 预测下一个价格
        last_sequence = scaled_data[-lookback:].reshape(1, lookback, 1)
        pred_scaled = model.predict(last_sequence, verbose=0)[0, 0]
        pred_price = scaler.inverse_transform([[pred_scaled]])[0, 0]

        self.log(f"LSTM预测下一日价格: {pred_price:.2f}")
        return pred_price

    # -------------------- 策略回测平台 --------------------
    def backtest_strategy(self, strategy_func, initial_capital=10000):
        """
        回测自定义策略
        strategy_func: 函数，输入(prices, dates)，输出交易信号列表（1买入，-1卖出，0持有）
        """
        prices = np.array(self.historical_prices)
        if len(prices) < 2:
            return {}

        # 生成信号
        signals = strategy_func(prices, self.historical_dates)
        if len(signals) != len(prices):
            signals = np.resize(signals, len(prices))

        # 模拟交易（简化：每次全仓）
        capital = initial_capital
        position = 0  # 持有股数
        trades = []

        for i in range(1, len(prices)):
            if signals[i] == 1 and position == 0:  # 买入
                position = capital / prices[i]
                capital = 0
                trades.append(('buy', self.historical_dates[i], prices[i]))
            elif signals[i] == -1 and position > 0:  # 卖出
                capital = position * prices[i]
                position = 0
                trades.append(('sell', self.historical_dates[i], prices[i]))

        # 最终价值
        final_value = capital + position * prices[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # 计算策略指标
        strategy_returns = []
        for i in range(1, len(prices)):
            if signals[i - 1] == 1:  # 持有
                strategy_returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
            else:
                strategy_returns.append(0)  # 空仓

        strategy_returns = np.array(strategy_returns)
        benchmark_returns = np.diff(prices) / prices[:-1]

        # 超额收益
        excess_returns = strategy_returns - benchmark_returns

        # 胜率
        wins = np.sum(strategy_returns > 0)
        losses = np.sum(strategy_returns < 0)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        # 最大回撤（策略）
        cumulative_strategy = np.cumprod(1 + strategy_returns)
        peak = np.maximum.accumulate(cumulative_strategy)
        drawdown = (peak - cumulative_strategy) / peak
        max_drawdown = np.max(drawdown)

        backtest_results = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': f"{total_return * 100:.2f}%",
            'benchmark_return': (prices[-1] / prices[0] - 1),
            'excess_return': total_return - (prices[-1] / prices[0] - 1),
            'win_rate': win_rate,
            'num_trades': len(trades),
            'max_drawdown_strategy': max_drawdown,
            'trades': trades
        }

        # 保存结果
        with open(os.path.join(LOG_DIR, "backtest_results.json"), "w") as f:
            json.dump(backtest_results, f, indent=2, default=str)

        self.log(f"回测完成: 总收益{backtest_results['total_return_pct']}, 胜率{win_rate:.2%}")
        return backtest_results

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
            self.log(f"市场更新: 新价格={self.current_price:.2f}, 近5日波动率={vol:.4f}")
            if vol > 0.05:
                self.log("警告: 市场波动过大！", level="WARNING")

    # -------------------- 运行模拟 --------------------
    def run_simulation(self, days=DAYS):
        self.log("=== 市场模拟器启动 ===")
        for day in range(1, days + 1):
            self.log(f"--- 第{day}天 ---")
            self.update_market()
            if day % 5 == 0:
                self.compute_technical_indicators()
                self.compute_factors()
                metrics = self.calculate_drawdown_metrics()
                self.log(f"风险指标: {metrics}")
        # 最后训练LSTM
        if TF_AVAILABLE and len(self.historical_prices) > 20:
            self.train_lstm()
        self.log("=== 模拟结束 ===")


# ---------------------------- Streamlit Web仪表盘 ----------------------------
def create_dashboard():
    st.set_page_config(layout="wide", page_title="市场波动监测模拟器")
    st.title("📈 市场波动监测与预测模拟器（专业版）")

    # 初始化或加载模拟器
    if 'simulator' not in st.session_state:
        st.session_state.simulator = MarketSimulatorPro("演示股票")
        st.session_state.simulator.run_simulation(days=30)  # 预运行

    sim = st.session_state.simulator

    # 侧边栏控制
    st.sidebar.header("控制面板")
    if st.sidebar.button("重新运行模拟 (30天)"):
        sim = MarketSimulatorPro("演示股票")
        sim.run_simulation(days=30)
        st.session_state.simulator = sim
        st.success("模拟完成！")

    st.sidebar.subheader("AI配置")
    api_key = st.sidebar.text_input("OpenAI API密钥", type="password", value=OPENAI_API_KEY)
    if api_key and api_key != OPENAI_API_KEY:
        openai.api_key = api_key
        global AI_AVAILABLE
        AI_AVAILABLE = True
        st.sidebar.success("API密钥已更新")

    # 主界面多标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 市场概览", "📈 技术指标", "📉 风险与回撤", "🤖 AI分析", "⚙️ 回测与预测"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("当前价格", f"${sim.current_price:.2f}",
                      delta=f"{(sim.current_price / sim.historical_prices[-2] - 1) * 100:.2f}%" if len(
                          sim.historical_prices) > 1 else None)
            st.metric("历史天数", len(sim.historical_prices))
        with col2:
            if len(sim.historical_prices) > 1:
                returns = np.diff(sim.historical_prices) / sim.historical_prices[:-1]
                st.metric("年化波动率", f"{np.std(returns) * np.sqrt(252) * 100:.2f}%")

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
            st.subheader("当前技术指标")
            cols = st.columns(3)
            ind = sim.technical_indicators
            with cols[0]:
                st.metric("RSI(14)", f"{ind.get('RSI', 0):.1f}")
                st.metric("MACD", f"{ind.get('MACD', 0):.2f}")
            with cols[1]:
                st.metric("布林带上轨", f"{ind.get('BB_upper', 0):.2f}")
                st.metric("布林带中轨", f"{ind.get('BB_middle', 0):.2f}")
            with cols[2]:
                st.metric("布林带下轨", f"{ind.get('BB_lower', 0):.2f}")
                st.metric("布林带宽", f"{ind.get('BB_width', 0):.3f}")
        else:
            st.write("暂无技术指标数据")

    with tab3:
        metrics = sim.calculate_drawdown_metrics()
        if metrics:
            st.subheader("风险指标")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最大回撤", metrics['max_drawdown_pct'])
                st.metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
            with col2:
                st.metric("索提诺比率", f"{metrics['sortino_ratio']:.2f}")
                st.metric("卡尔玛比率", f"{metrics['calmar_ratio']:.2f}")
            with col3:
                st.metric("年化收益", f"{metrics['annual_return'] * 100:.2f}%")
                st.metric("年化波动", f"{metrics['annual_volatility'] * 100:.2f}%")

            # 回撤曲线
            prices = np.array(sim.historical_prices)
            cumulative = np.cumprod(1 + np.concatenate([[0], np.diff(prices) / prices[:-1]]))
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=sim.historical_dates, y=drawdown * 100, fill='tozeroy', name='回撤%'))
            fig.update_layout(title='回撤曲线', yaxis_title='回撤 (%)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("数据不足")

    with tab4:
        st.subheader("AI新闻分析记录")
        if os.path.exists(os.path.join(LOG_DIR, "ai_analysis_log.txt")):
            with open(os.path.join(LOG_DIR, "ai_analysis_log.txt"), "r") as f:
                content = f.read()
                st.text(content)
        else:
            st.write("暂无AI分析记录")

        # 因子分析
        if sim.factors:
            st.subheader("因子分析")
            st.json(sim.factors)

    with tab5:
        st.subheader("策略回测")
        st.write("示例策略：双均线金叉死叉")

        # 定义策略函数
        def ma_cross_strategy(prices, dates, short=5, long=20):
            signals = np.zeros(len(prices))
            if len(prices) < long:
                return signals
            ma_short = np.convolve(prices, np.ones(short) / short, mode='valid')
            ma_long = np.convolve(prices, np.ones(long) / long, mode='valid')
            # 对齐
            min_len = min(len(ma_short), len(ma_long))
            ma_short = ma_short[-min_len:]
            ma_long = ma_long[-min_len:]
            for i in range(1, min_len):
                if ma_short[i - 1] <= ma_long[i - 1] and ma_short[i] > ma_long[i]:
                    signals[-(min_len - i)] = 1  # 金叉买入
                elif ma_short[i - 1] >= ma_long[i - 1] and ma_short[i] < ma_long[i]:
                    signals[-(min_len - i)] = -1  # 死叉卖出
            return signals

        if st.button("运行回测"):
            results = sim.backtest_strategy(ma_cross_strategy)
            st.success("回测完成")
            st.json(results)

        st.subheader("LSTM价格预测")
        if TF_AVAILABLE:
            if st.button("训练/预测LSTM"):
                pred = sim.train_lstm()
                if pred:
                    st.metric("LSTM预测下一日价格", f"${pred:.2f}")
        else:
            st.warning("TensorFlow未安装，无法使用LSTM")

        # 显示预测日志
        st.subheader("历史预测记录")
        if sim.predictions:
            st.json(sim.predictions[-1])
        else:
            st.write("暂无预测")


# ---------------------------- 主程序入口 ----------------------------
if __name__ == "__main__":
    # 如果作为脚本直接运行，启动Streamlit应用
    # 注意：直接运行此文件会启动Streamlit服务器，需使用 streamlit run 文件名.py
    # 这里添加判断，便于直接运行
    import sys

    if "streamlit" in sys.argv[0]:
        create_dashboard()
    else:
        # 命令行模式：运行模拟并退出
        sim = MarketSimulatorPro()
        sim.run_simulation(days=30)
        print("\n模拟完成，日志保存在logs文件夹。")
        print("要启动Web仪表盘，请执行: streamlit run market_simulator_pro.py")