from collections import OrderedDict
from typing import List, Tuple, Dict
import itertools
import random

from pypfopt import expected_returns, EfficientSemivariance, EfficientCVaR, EfficientCDaR, HRPOpt, BlackLittermanModel, black_litterman
from pypfopt.risk_models import CovarianceShrinkage, sample_cov, exp_cov
from pypfopt.risk_models import risk_matrix
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import *
import riskfolio as rp

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import *
from dateutil.relativedelta import relativedelta

def pickStocksByCorrelation(corr_matrix: pd.DataFrame, num_stocks: int, corr_type: str, balance_factor: float = 0.4):
    """
    Vybere podskupinu aktiv z korelační matice o zadané velikosti a korelačním typu.

    Parameters:
        corr_matrix (pd.DataFrame): Korelační matice vypočtená z denních výnosů aktiv.
        num_stocks (int): Počet požadovaných aktiv v podskupině.
        corr_type (str): Typ požadovaného korelačního typu portfolia 'high', 'mid', 'low' nebo 'mixed'.
                         - 'high': vrátí vysoce korelované portfolio.
                         - 'mid': vrátí portfolio s korelacemi okolo 0.5.
                         - 'low': vrátí málo korelované portfolio.
                         - 'mixed': vrátí portfolio, kde jsou málo i vysoce korelovaná aktiva.
        balance_factor (float): faktor proporce málo a vysoce korelovaných aktiv.

    Returns:
        list: Seznam vyfiltrovaných aktiv do portfolia.
        dict: Dodatečné informace pokud corr_type je 'mixed'.
    """
    if corr_type == 'high':
        threshold = 0.7
        condition = lambda x: x >= threshold
    elif corr_type == 'low':
        threshold = 0.3
        condition = lambda x: x <= threshold
    elif corr_type == 'mid':
        lower_threshold = 0.3
        upper_threshold = 0.7
        condition = lambda x: lower_threshold <= x <= upper_threshold
    elif corr_type == 'mixed':
        low_threshold = 0.3
        high_threshold = 0.7
        condition = lambda x: x <= low_threshold or x >= high_threshold
    else:
        raise ValueError("corr_type musí být 'high', 'mid', 'low', or 'mixed'")

    tickers = list(corr_matrix.columns)

    def is_valid_addition(group, candidate):
        for stock in group:
            corr = corr_matrix.loc[stock, candidate]
            if corr_type == 'mixed':
                if not (corr <= low_threshold or corr >= high_threshold):
                    return False
            else:
                if not condition(corr):
                    return False
        return True

    def get_correlation_stats(group):
        if len(group) < 2:
            return {"high_ratio": 0, "low_ratio": 0, "total_pairs": 0}

        pairs = list(itertools.combinations(group, 2))
        correlations = [corr_matrix.loc[s1, s2] for s1, s2 in pairs]

        high_count = sum(1 for c in correlations if c >= high_threshold)
        low_count = sum(1 for c in correlations if c <= low_threshold)
        mid_count = len(correlations) - high_count - low_count

        return {
            "high_count": high_count,
            "low_count": low_count,
            "mid_count": mid_count,
            "high_ratio": high_count / len(correlations) if correlations else 0,
            "low_ratio": low_count / len(correlations) if correlations else 0,
            "total_pairs": len(correlations)
        }

    def score_mixed_group(group):
        stats = get_correlation_stats(group)

        if stats["high_count"] == 0 or stats["low_count"] == 0:
            return -1000

        balance_score = -abs(stats["high_ratio"] - balance_factor) * 10

        extremeness_score = -stats["mid_count"] * 2

        return balance_score + extremeness_score

    if corr_type == 'mixed':
        best_group = []
        best_score = float('-inf')
        best_stats = {}

        num_attempts = min(50, len(tickers))
        seed_tickers = random.sample(tickers, num_attempts)

        for seed in seed_tickers:
            group = [seed]
            valid_candidates = [t for t in tickers if t != seed and is_valid_addition(group, t)]

            while len(group) < num_stocks and valid_candidates:
                best_candidate = None
                best_candidate_score = float('-inf')

                for candidate in valid_candidates[:]:
                    test_group = group + [candidate]

                    if len(test_group) == num_stocks:
                        score = score_mixed_group(test_group)
                        if score > best_candidate_score:
                            best_candidate = candidate
                            best_candidate_score = score
                    else:
                        remaining = [t for t in tickers if t not in test_group and is_valid_addition(test_group, t)]
                        score = len(remaining)
                        if score > best_candidate_score:
                            best_candidate = candidate
                            best_candidate_score = score

                if best_candidate:
                    group.append(best_candidate)
                    valid_candidates = [t for t in tickers if t not in group and is_valid_addition(group, t)]
                else:
                    break

            if len(group) == num_stocks:
                score = score_mixed_group(group)
                stats = get_correlation_stats(group)

                if score > best_score:
                    best_group = group
                    best_score = score
                    best_stats = stats

        if best_group:
            return best_group, best_stats
        return [], {}

    for seed in tickers:
        group = [seed]
        candidates = [t for t in tickers if t != seed]

        while len(group) < num_stocks and candidates:
            for candidate in candidates[:]:
                if is_valid_addition(group, candidate):
                    group.append(candidate)
                    candidates.remove(candidate)
                    break
                else:
                    candidates.remove(candidate)

            if len(group) == num_stocks:
                return group, {}

    return [], {}

def weightsPlot(weights, kind = "pie", title: str = "Weights"):
    match kind:
        case "pie": fig = go.Figure(data=[go.Pie(labels=list(weights.keys()), values=list(weights.values()))])
        case _ : raise(TypeError("Zatím jsou podporované pouze koláčové grafy 'pie'"))

    fig.update_layout(title_text= title, title_x=0.5, width = 700, height = 700)
    fig.show()

def weightsRebalancing(neutral_weights, returns_df, rebalance_freq='ME'):
    """
    Generuje DataFrame každodenních vah pro každé aktivum. 
    Dochází k pravidelnému rebalancingu podle zadané frekvence rebalancingu (rebalance_freq) a váhy se tak navrací k neutrálu.
    Například při měsíčním rebalancingu se váhy vrací k neitrálu ke konci každého měsíce, jinak se váhy odvíjí od denních výnosů.

    Parameters:
      neutral_weights (OrderedDict): OrderedDict, kde keys jsou aktiva a hodnoty jsou požadovaného neutrální váhy.
      returns_df (pd.DataFrame): DataFrame denních výnosů aktiv v portfoliu.
      rebalance_freq (str): Frekvence rebalancingu - 'D' pro denní, 'W' pro týdenní, 'ME' pro měsíční, 'QE' pro kvartální a 'YE' pro roční.

    Returns:
      pd.DataFrame: DataFrame of portfolio weights on the same index as returns_df.
    """
    returns_df = returns_df.sort_index()
    neutral = pd.Series(neutral_weights)
    weights_df = pd.DataFrame(index=returns_df.index, columns=returns_df.columns, dtype=float)
    
    rebal_dates = set(
        group.index[0] for _, group in returns_df.groupby(pd.Grouper(freq=rebalance_freq)) if not group.empty
    )
    
    rebal_dates.add(returns_df.index[0])
    current_weights = neutral.copy()
    weights_df.iloc[0] = current_weights
    
    for i in range(1, len(returns_df)):
        current_date = returns_df.index[i]
        
        if current_date in rebal_dates:
            current_weights = neutral.copy()
            weights_df.iloc[i] = current_weights
        else:
            daily_return = returns_df.iloc[i]
            updated = current_weights * (1 + daily_return)
            current_weights = updated / updated.sum()
            weights_df.iloc[i] = current_weights
            
    return weights_df


def MVOTangentPortfolio(prices, expected_returns_method = "historical_mean", cov_matrix_method = "sample", allow_shorting = False):
    match expected_returns_method:
        case "historical_mean": mu = expected_returns.mean_historical_return(prices)
        case "ema": mu = expected_returns.ema_historical_return(prices)
        case _: raise(ValueError("Očekávané výnosy musí být stanoveny historickým průměrem 'historical_mean' nebo exponencionálním průměrem 'ema'"))

    match cov_matrix_method:
        case "sample": S = sample_cov(prices)
        case "exp": S = exp_cov(prices)
        case "ledoit_cc": S = risk_matrix(prices, method="ledoit_wolf_constant_correlation")
        case "ledoit_cv": S = risk_matrix(prices, method="ledoit_wolf_constant_variance")
        case "ledoit_single_factor": S = risk_matrix(prices, method="ledoit_wolf_single_factor")
        case "oracle": S = risk_matrix(prices, method="oracle_approximating")
        case _: raise(ValueError("Kovariační matice musí být z metody 'sample', 'exp', 'ledoit_cc', 'ledoit_cv', 'ledoit_single_factor' nebo 'oracle'"))

    match allow_shorting:
        case False: ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        case True: ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

    weights = ef.max_sharpe()
    print(ef.portfolio_performance(verbose=True))
    return weights

def MVOPlotEfficientFrontier(prices, expected_returns_method = "historical_mean", cov_matrix_method = "sample", allow_shorting = False):
    match expected_returns_method:
        case "historical_mean": mu = expected_returns.mean_historical_return(prices)
        case "ema": mu = expected_returns.ema_historical_return(prices)
        case _: raise(ValueError("Očekávané výnosy musí být stanoveny historickým průměrem 'historical_mean' nebo exponencionálním průměrem 'ema'"))

    match cov_matrix_method:
        case "sample": S = sample_cov(prices)
        case "exp": S = exp_cov(prices)
        case "ledoit_cc": S = risk_matrix(prices, method="ledoit_wolf_constant_correlation")
        case "ledoit_cv": S = risk_matrix(prices, method="ledoit_wolf_constant_variance")
        case "ledoit_single_factor": S = risk_matrix(prices, method="ledoit_wolf_single_factor")
        case "oracle": S = risk_matrix(prices, method="oracle_approximating")
        case _: raise(ValueError("Kovariační matice musí být z metody 'sample', 'exp', 'ledoit_cc', 'ledoit_cv', 'ledoit_single_factor' nebo 'oracle'"))

    match allow_shorting:
        case False: ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        case True: ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

    return plot_efficient_frontier(opt = ef, points = 10000, show_assets=True, show_tickers=True)

def rpMVO(prices, risk_measure = "MV"):
    returns = prices.pct_change().dropna()
    port = rp.Portfolio(returns = returns)
    port.assets_stats(method_mu = "hist", method_cov = "hist")

    weights = port.optimization(model = "Classic", rm = risk_measure, obj = "Sharpe", rf = 0, l = 0, hist = True)
    return weights

def MVOTargetReturn(prices, target_return = 0.10, expected_returns_method = "historical_mean", cov_matrix_method = "sample", allow_shorting = False):
    match expected_returns_method:
        case "historical_mean": mu = expected_returns.mean_historical_return(prices)
        case "ema": mu = expected_returns.ema_historical_return(prices)
        case _: raise(ValueError("Očekávané výnosy musí být stanoveny historickým průměrem 'historical_mean' nebo exponencionálním průměrem 'ema'"))

    match cov_matrix_method:
        case "sample": S = sample_cov(prices)
        case "exp": S = exp_cov(prices)
        case "ledoit_cc": S = risk_matrix(prices, method="ledoit_wolf_constant_correlation")
        case "ledoit_cv": S = risk_matrix(prices, method="ledoit_wolf_constant_variance")
        case "ledoit_single_factor": S = risk_matrix(prices, method="ledoit_wolf_single_factor")
        case "oracle": S = risk_matrix(prices, method="oracle_approximating")
        case _: raise(ValueError("Kovariační matice musí být z metody 'sample', 'exp', 'ledoit_cc', 'ledoit_cv', 'ledoit_single_factor' nebo 'oracle'"))

    match allow_shorting:
        case False: ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        case True: ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

    weights = ef.efficient_return(target_return = target_return)
    print(ef.portfolio_performance(verbose = True))
    return weights

def RobustLedoitWolfCustomDeltaMaxSharpe(prices, expected_returns_method = "historical_mean", custom_delta = 0.4, allow_shorting = False):
    match expected_returns_method:
        case "historical_mean": mu = expected_returns.mean_historical_return(prices)
        case "ema": mu = expected_returns.ema_historical_return(prices)
        case _: raise(ValueError("Očekávané výnosy musí být stanoveny historickým průměrem 'historical_mean' nebo exponencionálním průměrem 'ema'"))

    S = CovarianceShrinkage(prices).shrunk_covariance(delta = custom_delta)

    match allow_shorting:
        case False: ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        case True: ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

    weights = ef.max_sharpe()
    print(ef.portfolio_performance(verbose = True))
    return weights

def RobustLedoitWolfCustomDeltaTargetReturn(prices, target_return = 0.10 ,expected_returns_method = "historical_mean", custom_delta = 0.4, allow_shorting = False):
    match expected_returns_method:
        case "historical_mean": mu = expected_returns.mean_historical_return(prices)
        case "ema": mu = expected_returns.ema_historical_return(prices)
        case _: raise(ValueError("Očekávané výnosy musí být stanoveny historickým průměrem 'historical_mean' nebo exponencionálním průměrem 'ema'"))

    S = CovarianceShrinkage(prices).shrunk_covariance(delta = custom_delta)

    match allow_shorting:
        case False: ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        case True: ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))

    weights = ef.efficient_return(target_return = target_return)
    print(ef.portfolio_performance(verbose=True))
    return weights

def RobustLedoitWolfCustomDeltaPlotEfficientFrontier(prices, expected_returns_method = "historical_mean", custom_delta = 0.4, allow_shorting = False):
    match expected_returns_method:
        case "historical_mean": mu = expected_returns.mean_historical_return(prices)
        case "ema": mu = expected_returns.ema_historical_return(prices)
        case _: raise(ValueError("Očekávané výnosy musí být stanoveny historickým průměrem 'historical_mean' nebo exponencionálním průměrem 'ema'"))
    
    S = CovarianceShrinkage(prices).shrunk_covariance(delta = custom_delta)
    
    match allow_shorting:
        case False: ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
        case True: ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    
    return plot_efficient_frontier(opt = ef, points = 10000, show_assets=True, show_tickers=True)

def CVaROptimizationMinCVAR(prices, expected_returns_method = "historical_mean", alpha = 0.05):
    match expected_returns_method:
        case "historical_mean": mu = expected_returns.mean_historical_return(prices)
        case "ema": mu = expected_returns.ema_historical_return(prices)
    
    hist_returns = expected_returns.returns_from_prices(prices)

    ef = EfficientCVaR(mu, hist_returns, beta = 1-alpha)
    ef.min_cvar()
    weights = ef.clean_weights()

    print(ef.portfolio_performance(verbose=True))
    return weights

def CVaROptimizationTargetReturn(prices, target_return = 0.10, expected_returns_method = "historical_mean", alpha = 0.05):
    match expected_returns_method:
        case "historical_mean": mu = expected_returns.mean_historical_return(prices)
        case "ema": mu = expected_returns.ema_historical_return(prices)
    
    hist_returns = expected_returns.returns_from_prices(prices)

    ef = EfficientCVaR(mu, hist_returns, beta = 1-alpha)
    ef.efficient_return(target_return = target_return)
    weights = ef.clean_weights()

    print(ef.portfolio_performance(verbose=True))
    return weights

def riskParityERC(prices, risk_measure = "MV"):
    returns = prices.pct_change().dropna()
    port = rp.Portfolio(returns = returns)
    port.assets_stats(method_mu="hist", method_cov="hist")
    weights = port.rp_optimization(model = "Classic", rm = risk_measure, rf = 0, b = None, hist = True)
    
    fig = plt.figure()
    rp.plot_risk_con(weights, cov = port.cov, returns = port.returns, rm = risk_measure, rf = 0, alpha = 0.05, height = 6, width = 10, ax = None)
    return OrderedDict(weights["weights"].to_dict()), fig

def HRPOptimization(prices):
    returns = prices.pct_change().dropna()
    hrp = HRPOpt(returns)
    weights = hrp.optimize()

    print(hrp.portfolio_performance(verbose=True))
    return weights

def blackLittermanAbsoluteViewsMaxSharpe(prices, views: OrderedDict):
    mu = expected_returns.mean_historical_return(prices)
    S = sample_cov(prices)

    spy_prices = yf.download('SPY', period = "10y", progress = False)['Close'].dropna()

    mcaps = {}

    for ticker in prices.columns:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap")
        mcaps[ticker] = market_cap


    delta = black_litterman.market_implied_risk_aversion(spy_prices)
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    picking = np.eye(len(prices.columns))

    blackLittermanModel = BlackLittermanModel(cov_matrix=S, absolute_views=views, pi=prior)
    rets_bl = blackLittermanModel.bl_returns()
    ef_bl = EfficientFrontier(rets_bl, S)
    weights = ef_bl.max_sharpe()

    print(ef_bl.portfolio_performance(verbose = True))
    return weights

def blackLittermanAbsoluteViewsTargetReturn(prices, views: OrderedDict, target_return):
    mu = expected_returns.mean_historical_return(prices)
    S = sample_cov(prices)

    spy_prices = yf.download('SPY', period = "10y", progress = False)['Close'].dropna()

    mcaps = {}

    for ticker in prices.columns:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap")
        mcaps[ticker] = market_cap

    delta = black_litterman.market_implied_risk_aversion(spy_prices)
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    picking = np.eye(len(prices.columns))

    blackLittermanModel = BlackLittermanModel(cov_matrix=S, absolute_views=views, pi=prior)
    rets_bl = blackLittermanModel.bl_returns()
    ef_bl = EfficientFrontier(rets_bl, S)
    weights = ef_bl.efficient_return(target_return = target_return)

    print(ef_bl.portfolio_performance(verbose=True))
    return weights

def blackLittermanAbsoluteViewsPlotEfficientFrontier(prices, views: OrderedDict):
    mu = expected_returns.mean_historical_return(prices)
    S = sample_cov(prices)

    spy_prices = yf.download('SPY', period = "10y", progress = False)['Close'].dropna()

    mcaps = {}

    for ticker in prices.columns:
        stock = yf.Ticker(ticker)
        market_cap = stock.info.get("marketCap")
        mcaps[ticker] = market_cap


    delta = black_litterman.market_implied_risk_aversion(spy_prices)
    prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)
    picking = np.eye(len(prices.columns))

    blackLittermanModel = BlackLittermanModel(cov_matrix=S, absolute_views=views, pi=prior)
    rets_bl = blackLittermanModel.bl_returns()
    ef_bl = EfficientFrontier(rets_bl, S)
    
    return plot_efficient_frontier(opt = ef_bl, points = 10000, show_assets=True, show_tickers=True)