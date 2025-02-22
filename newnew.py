# StockAnalysisSystem.py
import argparse
import time
import os
import efinance as ef
import akshare as ak
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.font_manager import FontProperties

# 添加修复多进程导入的代码
import multiprocessing
from multiprocessing.dummy import Pool

# ========== 系统配置 ==========
PUSH_TOKEN = 'f084c45f55ca4d658565498255db384b'
MIN_MV = 80
MAX_MV = 500
PE_THRESHOLD = 50
PB_THRESHOLD = 5
MACD_FAST = 8
MACD_SLOW = 20
MACD_SIGNAL = 7
VOLUME_THRESHOLD = 5e7
MAX_WORKERS = 3
REQUEST_INTERVAL = 0.5

# 评分权重配置
SCORE_WEIGHTS = {
    'technical': 0.35,    # 技术指标
    'chan': 0.30,         # 缠论信号
    'prediction': 0.25,   # 预测模型
    'risk': -0.10,        # 风险因素
}

# ========== 字体配置 ==========
font_path = os.path.join(os.path.dirname(__file__), 'SimHei.ttf')
if os.path.exists(font_path):
    custom_font = FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [custom_font.get_name()]
    plt.rcParams['axes.unicode_minus'] = False

# ========== 核心类定义 ==========
class StockAnalyzer:
    _main_board_cache = None

    @classmethod
    def get_main_board(cls):
        if cls._main_board_cache is None:
            retry_count = 0
            dynamic_volume = VOLUME_THRESHOLD
            dynamic_mv_min = MIN_MV
            
            while retry_count < 3:
                try:
                    df = ak.stock_zh_a_spot_em()
                    df['总市值'] = pd.to_numeric(df['总市值'], errors='coerce').fillna(0) / 1e8
                    df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0) * 100
                    df['代码'] = df['代码'].astype(str).str.zfill(6)
                    
                    filtered_df = df[
                        (df['代码'].str[:2].isin(['60', '00'])) &
                        (df['总市值'].between(dynamic_mv_min, MAX_MV)) &
                        (df['成交量'] > dynamic_volume)
                    ].copy()
                    
                    if not filtered_df.empty:
                        cls._main_board_cache = filtered_df[['代码', '名称']]
                        return cls._main_board_cache
                    else:
                        dynamic_volume = max(dynamic_volume * 0.7, 1e7)
                        dynamic_mv_min = max(dynamic_mv_min - 5, 10)
                        retry_count += 1
                        time.sleep(2)
                except Exception as e:
                    print(f"获取股票列表失败: {str(e)}")
                    time.sleep(3)
                    retry_count += 1
            cls._main_board_cache = pd.DataFrame()
        return cls._main_board_cache

    @classmethod
    def get_stock_name(cls, code):
        try:
            df = cls.get_main_board()
            if not df.empty:
                match = df[df['代码'] == code.zfill(6)]
                if not match.empty:
                    return match.iloc[0]['名称']
            return ef.stock.get_base_info(code).get('股票名称', '未知股票')
        except Exception as e:
            print(f"获取股票名称失败({code}): {str(e)}")
            return "未知股票"

    @staticmethod
    @lru_cache(maxsize=300)
    def get_enhanced_kline(code, period='daily'):
        try:
            start_date = (datetime.now() - timedelta(days=730)).strftime("%Y%m%d")
            df = ak.stock_zh_a_hist(symbol=code, period=period, adjust="qfq", start_date=start_date)
            if df.empty: return df
            
            df = df.rename(columns={'日期':'date','开盘':'open','收盘':'close','最高':'high','最低':'low','成交量':'volume'})
            df = df.sort_values('date', ascending=True).reset_index(drop=True)
            
            exp_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
            exp_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
            df['macd'] = exp_fast - exp_slow
            df['signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
            
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            return df
        except Exception as e:
            print(f"K线获取失败({code}): {str(e)}")
            return pd.DataFrame()

# ========== 缠论核心模块 ==========
class ChanTheoryAnalyzer:
    def __init__(self, data):
        self.data = data
        self.bi_list = []
        self.zhongshu_list = []

    def is_valid_fenxing(self, prev, curr, next_):
        # 顶分型：中间K线高点最高，且收盘价高于前一根的1.03倍
        is_top = (
            curr['high'] > prev['high'] * 1.03 and
            curr['high'] > next_['high'] * 1.03 and
            curr['close'] > prev['close'] * 1.015
        )
        # 底分型：中间K线低点最低，且收盘价低于前一根的0.985倍
        is_bottom = (
            curr['low'] < prev['low'] * 0.97 and
            curr['low'] < next_['low'] * 0.97 and
            curr['close'] < prev['close'] * 0.985
        )
        return is_top or is_bottom

    def detect_fenxing(self):
        fx_list = []
        for i in range(1, len(self.data)-1):
            prev = self.data.iloc[i-1]
            curr = self.data.iloc[i]
            next_ = self.data.iloc[i+1]
            if self.is_valid_fenxing(prev, curr, next_):
                fx_type = 'top' if curr['high'] > prev['high'] else 'bottom'
                fx_list.append({'type': fx_type, 'pos': i, 'price': curr['high'] if fx_type == 'top' else curr['low']})
        return fx_list

    def detect_bi(self):
        fx_list = self.detect_fenxing()
        bi_list = []
        prev_fx = None
        for curr_fx in fx_list:
            if prev_fx:
                if curr_fx['type'] != prev_fx['type']:
                    bi_type = '上升笔' if curr_fx['price'] > prev_fx['price'] else '下降笔'
                    bi_list.append({
                        'type': bi_type,
                        'start': {'date': self.data.iloc[prev_fx['pos']]['date'], 'price': prev_fx['price']},
                        'end': {'date': self.data.iloc[curr_fx['pos']]['date'], 'price': curr_fx['price']}
                    })
            prev_fx = curr_fx
        self.bi_list = bi_list
        return bi_list

    def detect_zhongshu(self):
        if len(self.bi_list) < 3:
            return []
        zhongshu_list = []
        for i in range(len(self.bi_list)-2):
            overlap_high = min(bi['end']['price'] for bi in self.bi_list[i:i+3])
            overlap_low = max(bi['start']['price'] for bi in self.bi_list[i:i+3])
            if overlap_high > overlap_low and (overlap_high - overlap_low)/overlap_low < 0.15:
                zhongshu_list.append({
                    'start': self.bi_list[i]['start']['date'],
                    'end': self.bi_list[i+2]['end']['date'],
                    'high': overlap_high,
                    'low': overlap_low
                })
        self.zhongshu_list = zhongshu_list
        return zhongshu_list

# ========== 预测系统 ==========
class StockPredictor:
    @staticmethod
    def get_stock_data(stock_code, start_date='20230101', end_date=datetime.now().strftime('%Y%m%d')):
        os.makedirs('results', exist_ok=True)
        file_path = f'results/{stock_code}.csv'
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            df = ef.stock.get_quote_history(stock_code, beg=start_date, end=end_date)
            if df.empty:
                tqdm.write(f"数据为空: {stock_code}")
                return pd.DataFrame()
            df.to_csv(file_path, index=False)
        
        df['日期'] = pd.to_datetime(df['日期'])
        cols = ['日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '换手率']
        df = df[cols].dropna()
        
        if len(df) < 100:
            tqdm.write(f"历史数据不足: {stock_code}")
            return pd.DataFrame()
        
        return df

    @staticmethod
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = -delta.where(delta < 0, 0).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def compute_bollinger_bands(series, window=20):
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        return sma + 2*std, sma - 2*std

    @staticmethod
    def create_features(df):
        df = df.copy()
        df['volume_price_ratio'] = df['成交量'].shift(1) / df['收盘'].shift(1)
        df['ma5'] = df['收盘'].rolling(5).mean().shift(1)
        df['ma20'] = df['收盘'].rolling(20).mean().shift(1)
        df['ma50'] = df['收盘'].rolling(50).mean().shift(1)
        df['rsi'] = StockPredictor.compute_rsi(df['收盘'], period=14).shift(1)
        df['boll_upper'], df['boll_lower'] = StockPredictor.compute_bollinger_bands(df['收盘'])
        return df.dropna()

    @staticmethod
    def prepare_data(df, test_size=0.2, forecast_horizon=5):
        df = StockPredictor.create_features(df)
        feature_columns = [col for col in df.columns if col not in ['收盘', '日期']]
        
        # 修改目标变量为未来第5天的收盘价（非移动平均）
        y = df['收盘'].shift(-forecast_horizon)
        valid_idx = y.dropna().index
        
        X = df.loc[valid_idx, feature_columns].astype(float)
        y = y.loc[valid_idx].astype(float).values.ravel()  # 强制转换为一维数组
        
        if len(X) == 0 or len(y) == 0:
            return None, None, None, None, None, None, None
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        dates = df.loc[valid_idx[split_idx:], '日期']
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_columns, dates

    @staticmethod
    def train_predict(stock_code):
        try:
            df = StockPredictor.get_stock_data(stock_code)
            if df.empty: return None

            X_train, X_test, y_train, y_test, scaler, features, dates = StockPredictor.prepare_data(df)
            
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 修复趋势判断逻辑
            last_close = df['收盘'].iloc[-1]
            trend = []
            for pred in y_pred:
                if pred > last_close * 1.03:
                    trend.append("🔥强烈看涨")
                elif pred > last_close:
                    trend.append("↑看涨")
                else:
                    trend.append("↓承压")

            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            r2 = r2_score(y_test, y_pred)
            
            # 修复绘图代码并关闭图形
            plt.figure(figsize=(12, 6))
            plt.plot(dates, y_test, label='实际价格', color='black')  # 移除 .values
            plt.plot(dates, y_pred, label='预测价格', linestyle='--', color='blue')
            plt.title(f'{stock_code} 五日趋势预测')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'results/{stock_code}_pred.png')
            plt.close()  # 显式关闭图形
            
            return {
                '代码': stock_code,
                'r2': r2,
                'mape': mape,
                'last_close': last_close,
                'pred_5day': y_pred[-1],
                '趋势': trend[-1]
            }
        except Exception as e:
            tqdm.write(f"预测失败({stock_code}): {str(e)}")
            return None

# ========== 主控制系统 ==========
def calculate_profit_potential(row):
    # 修改为安全访问方式
    weekly_bi_type = row.get('周线笔', {}).get('type', '') if row.get('周线笔') else ''
    tech_score = 0
    if weekly_bi_type == '上升笔':
        tech_score += 0.3
    if row['rsi'] < 70:
        tech_score += 0.2
        
    risk_penalty = 0
    if "周线下降笔" in row['风险']:
        risk_penalty -= 0.5
    if row['r2'] < 0.6:
        risk_penalty -= 0.3
        
    price_gain = (row['pred_5day'] - row['last_close']) / row['last_close']
    return price_gain * (1 + tech_score + risk_penalty)

def get_bi_info(bi_data, freq="日线"):
    if not bi_data or 'start' not in bi_data or 'end' not in bi_data:
        return f"{freq}：无有效笔信号"
    
    start_date_str = bi_data['start'].get('date', '')
    end_date_str = bi_data['end'].get('date', '')
    start_price = bi_data['start'].get('price', 0)
    end_price = bi_data['end'].get('price', 0)
    
    try:
        start_date = pd.to_datetime(start_date_str)
        end_date = pd.to_datetime(end_date_str)
        duration = (end_date - start_date).days
    except:
        return f"{freq}：日期数据异常"
    
    price_change = (end_price - start_price) / (start_price or 1) * 100
    return f"{freq}：{bi_data.get('type', '无')}运行{duration}天 | 幅度：{price_change:.1f}%"

def format_prediction(row):
    tags = []
    if row['r2'] > 0.7:
        tags.append("📊高可信度")
    if row['pred_5day'] > row['last_close'] * 1.03:
        tags.append("🔥强烈看涨")
    elif row['pred_5day'] > row['last_close']:
        tags.append("↑看涨")
    else:
        tags.append("↓承压")
    
    daily_signal = get_bi_info(row.get('日线笔'), "日线")
    weekly_signal = get_bi_info(row.get('周线笔'), "周线")
    
    risks = row.get('风险', [])
    risk_note = "⚠️ 风险提示：" + (" | ".join(risks) if risks else "无显著风险")
    
    return (
        f"{row['代码']} {row['名称']}\n"
        f"▸ 综合评分：{row['评分']:.1f} | 收益潜力：{row['收益潜力']:.1%}\n"
        f"▸ 五日均价预测：{row['pred_5day']:.2f}（现价：{row['last_close']:.2f}）\n"
        f"▸ 趋势判断：{row['趋势']} | 置信度：{row['r2']:.2f}\n"
        f"🔍 {daily_signal} | {weekly_signal}\n"
        f"{risk_note}\n"    
    )

def send_report(content):
    if PUSH_TOKEN and PUSH_TOKEN.strip():
        try:
            requests.post(
                "https://www.pushplus.plus/send",
                json={
                    "token": PUSH_TOKEN,
                    "title": "智能选股报告",
                    "content": content.replace('\n', '<br>'),
                    "template": "html"
                },
                timeout=10
            )
        except Exception as e:
            print(f"推送失败: {str(e)}")
            
# ========== 补充缺失的股票分析函数 ==========
def analyze_stock(code):
    try:
        daily_df = StockAnalyzer.get_enhanced_kline(code, period='daily')
        weekly_df = StockAnalyzer.get_enhanced_kline(code, period='weekly')
        
        if daily_df.empty or weekly_df.empty:
            return None

        # 计算50日均线
        daily_df['ma50'] = daily_df['close'].rolling(50).mean()

        # 技术指标分析
        rsi = StockPredictor.compute_rsi(daily_df['close']).iloc[-1]
        macd_status = "金叉" if daily_df['macd'].iloc[-1] > daily_df['signal'].iloc[-1] else "死叉"
        weekly_macd = weekly_df['macd'].iloc[-1]

        # 缠论分析
        daily_chan = ChanTheoryAnalyzer(daily_df)
        daily_bi = daily_chan.detect_bi()
        daily_bi = daily_bi[-1] if daily_bi else {}
        
        weekly_chan = ChanTheoryAnalyzer(weekly_df)
        weekly_bi = weekly_chan.detect_bi()
        weekly_bi = weekly_bi[-1] if weekly_bi else {}

        # 风险检测
        risks = []
        if rsi > 70:
            risks.append("RSI超买")
        if weekly_bi.get('type') == '下降笔':
            risks.append("周线下降笔")
        if daily_df['close'].iloc[-1] < daily_df['ma50'].iloc[-1]:
            risks.append("跌破年线")

        # 综合评分
        score = 7.0  # 基础分
        score += SCORE_WEIGHTS['technical'] * (1 if macd_status == "金叉" else -0.5)
        score += SCORE_WEIGHTS['chan'] * (1 if daily_bi.get('type') == '上升笔' else -0.5)
        score += SCORE_WEIGHTS['risk'] * len(risks)

        return {
            '代码': code,
            '名称': StockAnalyzer.get_stock_name(code),
            '评分': min(max(score, 0), 10),  # 评分限制在0-10分
            '风险': risks,
            'rsi': rsi,
            '日线笔': daily_bi,
            '周线笔': weekly_bi,
            '周线MACD': weekly_macd
        }
    except Exception as e:
        print(f"股票分析失败({code}): {str(e)}")
        return None

def main_controller(code=None, mode='strict'):
    print("\n=== 系统初始化检查 ===")
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if code:
        stock_list = [str(code).zfill(6)]
    else:
        base_df = StockAnalyzer.get_main_board()
        stock_list = base_df['代码'].tolist() if not base_df.empty else []
    
    print("\n=== 开始选股分析 ===")
    selected_stocks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(analyze_stock, code): code for code in stock_list}
        with tqdm(total=len(stock_list), desc="🔄 选股进度", unit="支") as pbar:
            for future in as_completed(futures):
                if result := future.result():
                    if result['评分'] >= 7 and len(result['风险']) < 2:
                        selected_stocks.append(result)
                pbar.update(1)

    print("\n=== 开始预测分析 ===")
    predictions = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(StockPredictor.train_predict, stock['代码']): stock for stock in selected_stocks}
        with tqdm(total=len(selected_stocks), desc="📊 预测进度", unit="支") as pbar:
            for future in as_completed(futures):
                stock_info = futures[future]
                if pred := future.result():
                    pred.update({
                        '名称': stock_info['名称'],
                        '评分': stock_info['评分'],
                        '风险': stock_info['风险'],
                        '日线笔': stock_info['日线笔'],
                        '周线笔': stock_info['周线笔'],
                        'rsi': stock_info['rsi'],
                        '周线MACD': stock_info['周线MACD']
                    })
                    predictions.append(pred)
                pbar.update(1)

    if predictions:
        df = pd.DataFrame(predictions)
        df['收益潜力'] = df.apply(calculate_profit_potential, axis=1)
        
        # 根据模式设置筛选条件
        if mode == 'strict':
            filtered = df[
                (df['收益潜力'] > 0) &
                (df['rsi'].between(30, 65)) &
                (df['周线MACD'] > 0)
            ]
        elif mode == 'aggressive':
            filtered = df[
                (df['收益潜力'] > 0.2) &
                (df['rsi'] < 70) &
                (df['周线MACD'] > -0.5)
            ]
        else:  # balanced
            filtered = df[
                (df['收益潜力'] > 0.1) &
                (df['rsi'].between(30, 70)) &
                (df['周线MACD'] > -0.2)
            ]
        
        top_5 = filtered.sort_values(
            by=['收益潜力', '评分'],
            ascending=[False, False]
        ).head(5)

        report_lines = [format_prediction(row) for _, row in top_5.iterrows()]
        
        print("\n=== 最终推荐股票 ===")
        print("\n".join(report_lines))
        
        # 添加调试信息
        print("\n=== 调试信息 ===")
        print("总预测记录数:", len(df))
        print("严格模式筛选条件:")
        print("收益潜力 > 0 的数量:", len(df[df['收益潜力'] > 0]))
        print("RSI在30-65的数量:", len(df[df['rsi'].between(30,65)]))
        print("周线MACD>0的数量:", len(df[df['周线MACD'] > 0]))
        print("交集数量:", len(filtered))
        
        report = f"""
📈 智能选股报告 {datetime.now().strftime('%Y-%m-%d')}
========================================
{"<br>".join(report_lines)}

💡 操作建议：
1. 优先选择带「📊高可信度」的标的
2. 出现「🔥强烈看涨」时建议重点跟踪
3. 结合30分钟K线MACD金叉寻找买点
4. 当出现「↓承压」时需谨慎追高
"""
        send_report(report)
    else:
        print("没有符合条件的股票可供推荐。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能选股预测系统")
    parser.add_argument("--code", type=str, help="指定股票代码（如：600000）")
    parser.add_argument("--mode", type=str, choices=['strict', 'aggressive', 'balanced'], default='strict', help="选股模式")
    args = parser.parse_args()
    
    main_controller(code=args.code, mode=args.mode)
