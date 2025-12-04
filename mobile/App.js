// mobile/App.js
import React, {
  useEffect,
  useState,
  useCallback,
  useMemo,
} from "react";

import {
  SafeAreaView,
  View,
  Text,
  ScrollView,
  Pressable,
  ActivityIndicator,
  RefreshControl,
  StyleSheet,
  Dimensions,
  Platform,
} from "react-native";
import axios from "axios";
import { LinearGradient } from "expo-linear-gradient";
import Svg, {
  Path,
  Defs,
  LinearGradient as SvgGradient,
  Stop,
} from "react-native-svg";
import { Feather } from "@expo/vector-icons";
import Constants from "expo-constants";

// ===========================
// CONFIG: BASE URL API
// ===========================

const resolveApiBaseUrl = () => {
  const envUrl =
    process.env.EXPO_PUBLIC_API_URL ||
    Constants.expoConfig?.extra?.API_BASE_URL ||
    Constants.manifest2?.extra?.API_BASE_URL ||
    Constants.manifest?.extra?.API_BASE_URL;

  if (envUrl) {
    return envUrl;
  }

  if (Platform.OS === "web") {
    return "http://localhost:8000";
  }

  // put actual IP from Expo here
  return "http://10.87.14.92:8000";
};

const API_BASE_URL = resolveApiBaseUrl();
console.log("StockAI API_BASE_URL =", API_BASE_URL);

// chart dimensions
const { width: SCREEN_WIDTH } = Dimensions.get("window");
const CHART_WIDTH = SCREEN_WIDTH - 32;
const CHART_HEIGHT = 180;

// Asset class -> code
const ASSET_CLASS_CODE = {
  BTC: "crypto",
  ETH: "crypto",
  TSLA: "us_stock",
  AAPL: "us_stock",
};

// ===========================
// I18N
// ===========================

const I18N = {
  ru: {
    "header.subtitle": "Прогнозы T+N на основе ML",
    "status.online": "Онлайн",

    "tabs.signals": "Сигналы",
    "tabs.backtest": "Бэктест",
    "tabs.about": "О продукте",

    "signals.loading": "Загрузка сигналов...",
    "signals.error": "Не удалось загрузить сигналы",

    "asset.classLabel": "Класс актива:",
    "asset.crypto": "Криптовалюта",
    "asset.us_stock": "Акция US рынка",
    "asset.default": "Финансовый актив",

    "chart.noData": "Недостаточно данных для графика",

    "prob.t1.title": "Вероятность роста T+1",

    "inline.buyCount": "Сильных BUY по портфелю",
    "inline.avgUpProb": "Средняя вероятность роста",

    "metrics.probability": "Вероятность роста",
    "metrics.confidence": "Уверенность сигнала",
    "metrics.predDate": "Дата прогноза",
    "metrics.horizon": "Горизонт",
    "metrics.horizonValue": "T+1 (следующий день)",

    "confidence.high": "Высокая",
    "confidence.medium": "Средняя",
    "confidence.low": "Низкая",

    "scenario.buy": "Сценарий: открыть лонг",
    "scenario.caution": "Сценарий: снизить риск",
    "scenario.neutral": "Сценарий: наблюдать",

    "list.allSignals": "Все сигналы",
    "list.change24h": "24ч изменение",

    "backtest.title": "Бэктест моделей",
    "backtest.subtitle":
      "Метрики на отложенном периоде: как T+1 (Long-only) стратегия ведёт себя против Buy&Hold по каждому активу.",
    "backtest.loading": "Загрузка метрик...",
    "backtest.error": "Не удалось загрузить метрики моделей",
    "backtest.metric.testPeriod": "Период теста",

    // Новые ключи backtest
    "backtest.metric.balAcc": "Balanced Accuracy",
    "backtest.metric.auc": "AUC",
    "backtest.metric.posRate": "Доля LONG сигналов",
    "backtest.metric.strategyReturn": "Доходность стратегии",
    "backtest.metric.buyHoldReturn": "Доходность Buy&Hold",
    "backtest.metric.excessReturn": "Excess Return",
    "backtest.metric.sharpe": "Sharpe (long-only)",
    "backtest.metric.sharpeConf": "Sharpe (confident)",
    "backtest.metric.winRate": "Win Rate (executed)",
    "backtest.metric.winRateConf": "Win Rate (confident)",
    "backtest.metric.coverage": "Coverage (gate)",
    "backtest.metric.tradesAll": "Сделок (всего)",
    "backtest.metric.tradesConf": "Сделок (confident)",

    "backtest.tag.modelBeatsBH": "Модель > Buy&Hold",
    "backtest.tag.bhBeatsModel": "Buy&Hold > Модель",
    "backtest.tag.mixed": "Смешанный результат",

    "pred.loading": "Обновляем прогноз для",

    // ABOUT
    "about.title": "О продукте",
    "about.main":
      "Это учебное приложение, которое показывает сигналы модели машинного обучения по активам BTC, ETH, TSLA и AAPL. Модель оценивает вероятность того, что цена закрытия следующего дня будет выше текущей.",
    "about.assetsTitle": "Какие активы сейчас поддерживаются",
    "about.assetsText":
      "• BTC — Bitcoin (криптовалюта)\n• ETH — Ethereum (криптовалюта)\n• TSLA — Tesla, Inc. (акция США)\n• AAPL — Apple Inc. (акция США)\n\nДальнейшие версии могут добавить новые активы и классы — форекс, индексы и т.п.",
    "about.t1Title": "Что такое T+1 / T+3 / T+7",
    "about.t1Text":
      "T+1, T+3 и T+7 — это горизонты прогноза «один день», «три дня» и «неделя» вперёд. Модели смотрят на исторические котировки и технические признаки и оценивают вероятность события:\n\n• T+1: Close(t+1) > Close(t)\n• T+3: Close(t+3) > Close(t)\n• T+7: Close(t+7) > Close(t)\n\nМодели не пытаются назвать точную цену, а оценивают направление движения на заданном горизонте.",
    "about.readSignalsTitle": "Как читать сигналы",
    "about.readSignalsText":
      "«Покупай» — по историческим данным модель видит повышенную вероятность роста на выбранном горизонте.\n«Осторожно» — вероятность роста низкая или неустойчивая, сигнал скорее защитный.\n«Нейтрально» — около 50/50, выраженного перекоса нет.\n\nПроцент вероятности — это оценка модели на основе бэктеста, а не гарантия результата.",
    "about.trainingTitle": "Как обучались модели",
    "about.trainingText":
      "• Исторические данные: несколько лет котировок по каждому активу.\n• Целевая переменная: бинарное событие роста на горизонте T+N.\n• Фичи: технические индикаторы (RSI, EMA, MACD, Bollinger Bands, ATR), лаги цен и объёмов, волатильность, календарные признаки.\n• Модели: XGBoost / градиентный бустинг с калибровкой вероятности.\n• Валидация: сплит по времени, отдельный holdout-период, контроль утечек и переобучения.",
    "about.limitationsTitle": "Ограничения и дисклеймер",
    "about.limitationsText":
      "• Инференс оффлайн: прогнозы считаются заранее и раздаются через API, модель не торгует в реальном времени.\n• Комиссии, спрэды, проскальзывание и налоги в расчётах не учтены.\n• Историческая доходность не гарантирует будущих результатов.\n• Приложение создано в учебных целях и не является инвестиционной рекомендацией или призывом к совершению сделок.",
    "about.howToUseTitle": "Как использовать сигналы в своей стратегии",
    "about.howToUseText":
      "1. Смотрите на вероятность, а не на цвет ярлыка: 52% и 68% — принципиально разные уровни.\n2. Используйте сигналы как дополнительный слой к своему анализу, а не единственный триггер входа.\n3. Не масштабируйтесь агрессивно от одного прогноза: отслеживайте поведение стратегии на серии сделок.\n4. Учитывайте риск-профиль, лимиты по позиции и общую волатильность актива.\n5. Регулярно сверяйте текущие результаты с бэктестом в разделе Backtest.",
    "about.pipelineTitle": "Как устроен pipeline",
    "about.pipelineText":
      "• Источник данных: исторические котировки BTC, ETH, TSLA и AAPL.\n• Фичи: технические индикаторы (RSI, EMA, MACD, Bollinger Bands, ATR), волатильность, лаги, объёмы, циклическое время (день недели, время).\n• Модель: градиентный бустинг (XGBoost) с калибровкой вероятностей.\n• Валидация: сплит по времени, отдельный holdout, метрики Balanced Accuracy, AUC, Strategy Return, Sharpe и Win Rate.",
    "about.techTitle": "Технологический стек",
    "about.techText":
      "Фронтенд: React Native / Expo с кастомной отрисовкой графиков.\nБэкенд: Python API (FastAPI / Uvicorn) с отдельным модулем для оффлайн-инференса и онлайн-дозагрузки свежих данных.\nML: XGBoost, scikit-learn, собственный пайплайн фичей и валидации.\nИнфраструктура: REST API, конфиги окружений, поддержка нескольких горизонтов.",
    "about.faqTitle": "FAQ — частые вопросы",
    "about.faqText":
      "Q: Могу ли я торговать только по этим сигналам?\nA: Это учебный инструмент. Используйте его как экспериментальный индикатор, а не как единственный источник решений.\n\nQ: Почему прогноз иногда не совпадает с фактом?\nA: Модель даёт вероятностную оценку, а не детерминированный исход. Даже при 70% шансы на неуспех — 30%.\n\nQ: Как часто обновляются данные?\nA: Котировки подгружаются регулярно, но инфраструктура рассчитана на дневной горизонт, а не на высокочастотный трейдинг.",
  },
  tr: {
    "header.subtitle": "ML tabanlı T+N tahminleri",
    "status.online": "Çevrimiçi",

    "tabs.signals": "Sinyaller",
    "tabs.backtest": "Backtest",
    "tabs.about": "Ürün hakkında",

    "signals.loading": "Sinyaller yükleniyor...",
    "signals.error": "Sinyaller alınamadı",

    "asset.classLabel": "Varlık sınıfı:",
    "asset.crypto": "Kripto para",
    "asset.us_stock": "ABD hisse senedi",
    "asset.default": "Finansal varlık",

    "chart.noData": "Grafik için yeterli veri yok",

    "prob.t1.title": "T+1 yükseliş olasılığı",

    "inline.buyCount": "Portföydeki güçlü BUY sinyalleri",
    "inline.avgUpProb": "Ortalama yükseliş olasılığı",

    "metrics.probability": "Yükseliş olasılığı",
    "metrics.confidence": "Sinyal güveni",
    "metrics.predDate": "Tahmin tarihi",
    "metrics.horizon": "Vade",
    "metrics.horizonValue": "T+1 (ertesi gün)",

    "confidence.high": "Yüksek",
    "confidence.medium": "Orta",
    "confidence.low": "Düşük",

    "scenario.buy": "Senaryo: long aç",
    "scenario.caution": "Senaryo: riski azalt",
    "scenario.neutral": "Senaryo: izle",

    "list.allSignals": "Tüm sinyaller",
    "list.change24h": "24s değişim",

    "backtest.title": "Modellerin backtest sonuçları",
    "backtest.subtitle":
      "Her varlık için T+1 (Long-only) stratejisinin basit Buy&Hold yaklaşımına karşı ertelenmiş dönemde nasıl davrandığını gösterir.",
    "backtest.loading": "Metrikler yükleniyor...",
    "backtest.error": "Backtest metrikleri alınamadı",
    "backtest.metric.testPeriod": "Test dönemi",

    "backtest.metric.balAcc": "Balanced Accuracy",
    "backtest.metric.auc": "AUC",
    "backtest.metric.posRate": "LONG oranı",
    "backtest.metric.strategyReturn": "Strateji getirisi",
    "backtest.metric.buyHoldReturn": "Buy&Hold getirisi",
    "backtest.metric.excessReturn": "Excess Return",
    "backtest.metric.sharpe": "Sharpe (long-only)",
    "backtest.metric.sharpeConf": "Sharpe (confident)",
    "backtest.metric.winRate": "Win Rate (işlem)",
    "backtest.metric.winRateConf": "Win Rate (confident)",
    "backtest.metric.coverage": "Coverage (gate)",
    "backtest.metric.tradesAll": "İşlem sayısı",
    "backtest.metric.tradesConf": "İşlem sayısı (conf)",

    "backtest.tag.modelBeatsBH": "Model > Buy&Hold",
    "backtest.tag.bhBeatsModel": "Buy&Hold > Model",
    "backtest.tag.mixed": "Karışık sonuç",

    "pred.loading": "Tahmin güncelleniyor:",

    // ABOUT
    "about.title": "Ürün hakkında",
    "about.main":
      "Bu, BTC, ETH, TSLA ve AAPL varlıkları için makine öğrenimi tabanlı sinyaller gösteren eğitim amaçlı bir uygulamadır. Model, ertesi günkü kapanış fiyatının bugünkü kapanıştan yüksek olma olasılığını tahmin eder.",
    "about.assetsTitle": "Şu anda desteklenen varlıklar",
    "about.assetsText":
      "• BTC — Bitcoin (kripto para)\n• ETH — Ethereum (kripto para)\n• TSLA — Tesla, Inc. (ABD hissesi)\n• AAPL — Apple Inc. (ABD hissesi)\n\nİlerleyen sürümlerde forex, endeks gibi yeni varlık sınıfları eklenebilir.",
    "about.t1Title": "T+1 / T+3 / T+7 nedir?",
    "about.t1Text":
      "T+1, T+3 ve T+7, sırasıyla bir iş günü, üç gün ve bir haftalık tahmin ufkunu ifade eder. Modeller, tarihsel fiyatlara ve teknik göstergelere bakarak aşağıdaki olayların olasılığını hesaplar:\n\n• T+1: Close(t+1) > Close(t)\n• T+3: Close(t+3) > Close(t)\n• T+7: Close(t+7) > Close(t)\n\nModel tam fiyatı değil, seçilen vade için yön olasılığını tahmin eder.",
    "about.readSignalsTitle": "Sinyaller nasıl okunur?",
    "about.readSignalsText":
      "\"Al\" — tarihsel verilere göre model, seçilen vadede yükseliş olasılığını yüksek görür.\n\"Dikkat\" — yükseliş olasılığı düşük veya kararsızdır, sinyal daha çok korumacı niteliktedir.\n\"Nötr\" — yaklaşık 50/50, belirgin bir üstünlük yoktur.\n\nYüzdelik olasılık, backtest sonuçlarına dayalı istatistiksel bir tahmindir; sonuç garantisi değildir.",
    "about.trainingTitle": "Modeller nasıl eğitildi?",
    "about.trainingText":
      "• Tarihsel veri: Her varlık için birkaç yıllık fiyat serileri.\n• Hedef değişken: T+N ufkunda fiyatın yükselip yükselmediğini gösteren ikili etiket.\n• Özellikler: teknik indikatörler (RSI, EMA, MACD, Bollinger Bands, ATR), fiyat ve hacim lagları, volatilite, takvimsel özellikler.\n• Modeller: Olasılık kalibrasyonlu XGBoost / gradient boosting.\n• Doğrulama: Zaman serisi split, ayrı holdout dönem, veri kaçağı ve aşırı öğrenme kontrolleri.",
    "about.limitationsTitle": "Sınırlamalar ve feragat",
    "about.limitationsText":
      "• Çevrimdışı inference: tahminler önceden hesaplanır ve API üzerinden sunulur, model gerçek zamanlı işlem yapmaz.\n• Komisyonlar, spreadler, slipaj ve vergiler hesaplamaya dahil edilmemiştir.\n• Geçmiş getiriler gelecekteki performansı garanti etmez.\n• Uygulama eğitim amaçlıdır; yatırım tavsiyesi veya işlem çağrısı değildir.",
    "about.howToUseTitle": "Sinyalleri stratejinize nasıl entegre edebilirsiniz?",
    "about.howToUseText":
      "1. Yalnızca etiketlere değil, olasılığın seviyesine bakın: %52 ile %68 aynı risk profiline sahip değildir.\n2. Sinyalleri tek karar kaynağı olarak değil, mevcut analizinizin üzerine gelen ek bir katman olarak düşünün.\n3. Tek bir tahmine göre agresif pozisyon almak yerine, seri işlem sonuçlarını izleyin.\n4. Kendi risk profiliniz, pozisyon limitleriniz ve varlığın volatilitesini mutlaka dikkate alın.\n5. Backtest sekmesindeki metriklerle canlı sonuçları düzenli olarak kıyaslayın.",
    "about.pipelineTitle": "Pipeline nasıl çalışıyor?",
    "about.pipelineText":
      "• Veri kaynağı: BTC, ETH, TSLA ve AAPL için tarihsel fiyat serileri.\n• Özellikler: teknik indikatörler (RSI, EMA, MACD, Bollinger Bands, ATR), volatilite, laglar, hacim ve döngüsel zaman (hafta günü, saat).\n• Model: olasılık kalibrasyonlu gradient boosting (XGBoost).\n• Doğrulama: zaman serisi split, ayrı holdout seti, Balanced Accuracy, AUC, Strategy Return, Sharpe ve Win Rate metrikleri.",
    "about.techTitle": "Teknoloji altyapısı",
    "about.techText":
      "Frontend: Grafiklerin özel çizildiği React Native / Expo.\nBackend: Offline inference ve canlı veri yenilemesi için Python tabanlı API (FastAPI / Uvicorn).\nML: XGBoost, scikit-learn, özel özellik ve doğrulama pipeline’ı.\nAltyapı: REST API, çoklu vade desteği, ortam konfigürasyonları.",
    "about.faqTitle": "SSS — sık sorulan sorular",
    "about.faqText":
      "S: Sadece bu sinyallere bakarak işlem yapabilir miyim?\nC: Bu araç eğitim amaçlıdır. Sinyalleri tek başına karar mekanizması olarak değil, ek bir gösterge olarak değerlendirin.\n\nS: Neden tahminler bazen gerçekleşen fiyatla uyuşmuyor?\nC: Model, deterministik değil, olasılık temelli sonuç üretir. %70 bile olsa, başarısızlık ihtimali %30’dur.\n\nS: Veriler ne kadar sık güncelleniyor?\nC: Fiyat verileri düzenli çekilir, ancak altyapı yüksek frekanslı işlem değil, günlük ufuk için tasarlanmıştır.",
  },
  en: {
    "header.subtitle": "T+N forecasts powered by ML",
    "status.online": "Online",

    "tabs.signals": "Signals",
    "tabs.backtest": "Backtest",
    "tabs.about": "About",

    "signals.loading": "Loading signals...",
    "signals.error": "Failed to load signals",

    "asset.classLabel": "Asset class:",
    "asset.crypto": "Cryptocurrency",
    "asset.us_stock": "US stock",
    "asset.default": "Financial asset",

    "chart.noData": "Not enough data to draw the chart",

    "prob.t1.title": "T+1 upside probability",

    "inline.buyCount": "Strong BUY signals in portfolio",
    "inline.avgUpProb": "Average upside probability",

    "metrics.probability": "Upside probability",
    "metrics.confidence": "Signal confidence",
    "metrics.predDate": "Forecast date",
    "metrics.horizon": "Horizon",
    "metrics.horizonValue": "T+1 (next day)",

    "confidence.high": "High",
    "confidence.medium": "Medium",
    "confidence.low": "Low",

    "scenario.buy": "Scenario: open long",
    "scenario.caution": "Scenario: reduce risk",
    "scenario.neutral": "Scenario: watch",

    "list.allSignals": "All signals",
    "list.change24h": "24h change",

    "backtest.title": "Backtest metrics",
    "backtest.subtitle":
      "Holdout-period metrics: how the T+1 (long-only) strategy performs versus simple Buy&Hold for each asset.",
    "backtest.loading": "Loading metrics...",
    "backtest.error": "Failed to load backtest metrics",
    "backtest.metric.testPeriod": "Test period",

    "backtest.metric.balAcc": "Balanced Accuracy",
    "backtest.metric.auc": "AUC",
    "backtest.metric.posRate": "LONG share",
    "backtest.metric.strategyReturn": "Strategy Return",
    "backtest.metric.buyHoldReturn": "Buy&Hold Return",
    "backtest.metric.excessReturn": "Excess Return",
    "backtest.metric.sharpe": "Sharpe (long-only)",
    "backtest.metric.sharpeConf": "Sharpe (confident)",
    "backtest.metric.winRate": "Win Rate (executed)",
    "backtest.metric.winRateConf": "Win Rate (confident)",
    "backtest.metric.coverage": "Coverage (gate)",
    "backtest.metric.tradesAll": "Trades (all)",
    "backtest.metric.tradesConf": "Trades (confident)",

    "backtest.tag.modelBeatsBH": "Model > Buy&Hold",
    "backtest.tag.bhBeatsModel": "Buy&Hold > Model",
    "backtest.tag.mixed": "Mixed result",

    "pred.loading": "Updating forecast for",

    // ABOUT
    "about.title": "About",
    "about.main":
      "This is an educational app that shows machine learning–based signals for BTC, ETH, TSLA and AAPL. The model estimates the probability that tomorrow’s close will be higher than today’s close.",
    "about.assetsTitle": "Which assets are supported?",
    "about.assetsText":
      "• BTC — Bitcoin (crypto)\n• ETH — Ethereum (crypto)\n• TSLA — Tesla, Inc. (US stock)\n• AAPL — Apple Inc. (US stock)\n\nFuture iterations may add new asset classes such as FX or indices.",
    "about.t1Title": "What is T+1 / T+3 / T+7",
    "about.t1Text":
      "T+1, T+3 and T+7 are one–day, three–day and one–week forecast horizons. The models look at historical prices and technical features and estimate the probability of:\n\n• T+1: Close(t+1) > Close(t)\n• T+3: Close(t+3) > Close(t)\n• T+7: Close(t+7) > Close(t)\n\nThey do not try to predict the exact future price, but the direction on the selected horizon.",
    "about.readSignalsTitle": "How to read the signals",
    "about.readSignalsText":
      "\"Buy\" — based on historical data the model sees a higher probability of upside on the selected horizon.\n\"Caution\" — upside probability is low or unstable, the signal is defensive.\n\"Neutral\" — roughly 50/50, no strong edge.\n\nThe probability in % is a model estimate from backtests, not a guarantee of future performance.",
    "about.trainingTitle": "How the models were trained",
    "about.trainingText":
      "• Historical data: several years of price history per asset.\n• Target: binary label indicating whether the price increased on horizon T+N.\n• Features: technical indicators (RSI, EMA, MACD, Bollinger Bands, ATR), price and volume lags, volatility, calendar features.\n• Models: XGBoost / gradient boosting with probability calibration.\n• Validation: time-series split with a separate holdout period, leakage checks and overfitting control.",
    "about.limitationsTitle": "Limitations & disclaimer",
    "about.limitationsText":
      "• Offline inference: forecasts are precomputed and served via API; the model does not trade in real time.\n• Commissions, spreads, slippage and taxes are not included.\n• Past performance does not guarantee future results.\n• This app is for educational purposes only and is not investment advice or a solicitation to trade.",
    "about.howToUseTitle": "How to use the signals in practice",
    "about.howToUseText":
      "1. Focus on probability levels, not just the label: 52% and 68% represent very different risk profiles.\n2. Treat the model as an additional layer on top of your own analysis, not a single source of truth.\n3. Avoid scaling aggressively off a single prediction; monitor performance over a series of trades.\n4. Respect your risk limits, position sizing rules and the asset’s volatility.\n5. Regularly compare live behavior to the backtest metrics in the Backtest tab.",
    "about.pipelineTitle": "How the pipeline works",
    "about.pipelineText":
      "• Data source: historical prices for BTC, ETH, TSLA and AAPL.\n• Features: technical indicators (RSI, EMA, MACD, Bollinger Bands, ATR), volatility, lags, volume and cyclical time (day of week, time).\n• Model: gradient boosting (XGBoost) with probability calibration.\n• Validation: time–series split with a separate holdout set; metrics include Balanced Accuracy, AUC, Strategy Return, Sharpe and Win Rate.",
    "about.techTitle": "Technology stack",
    "about.techText":
      "Frontend: React Native / Expo with custom chart rendering.\nBackend: Python API (FastAPI / Uvicorn) with a separate module for offline inference and online data refresh.\nML: XGBoost, scikit-learn and a custom feature/validation pipeline.\nInfra: REST API, multiple horizons, environment-based configuration.",
    "about.faqTitle": "FAQ — frequently asked questions",
    "about.faqText":
      "Q: Can I trade purely based on these signals?\nA: This is an educational tool. Use it as an experimental indicator, not as your only decision source.\n\nQ: Why do forecasts sometimes disagree with reality?\nA: The model outputs probabilities, not certainties. Even with 70% probability, there is a 30% chance of the opposite outcome.\n\nQ: How often is data refreshed?\nA: Quotes are updated regularly, but the infrastructure is designed for daily horizons, not high-frequency trading.",
  },
};

const translate = (lang, key) =>
  I18N[lang]?.[key] ?? I18N["ru"][key] ?? key;

// Verdicts come from the backend in Russian — map them to labels
const VERDICT_LABELS = {
  ru: {
    "Покупай": "Покупай",
    "Осторожно": "Осторожно",
    "Нейтрально": "Нейтрально",
  },
  tr: {
    "Покупай": "Al",
    "Осторожно": "Dikkat",
    "Нейтрально": "Nötr",
  },
  en: {
    "Покупай": "Buy",
    "Осторожно": "Caution",
    "Нейтрально": "Neutral",
  },
};

// Threshold logic for confidence — compute code, then translate
const computeConfidenceCode = (p) => {
  const diff = Math.abs((p ?? 0.5) - 0.5);
  if (diff >= 0.15) return "high";
  if (diff >= 0.08) return "medium";
  return "low";
};

// ===========================
// HELPERS
// ===========================

const formatUsd = (value) => {
  if (typeof value !== "number") return "—";
  try {
    return (
      "$" +
      value.toLocaleString("en-US", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })
    );
  } catch {
    return "$" + value.toFixed(2);
  }
};

const formatPercent = (value, digits = 1) => {
  if (typeof value !== "number") return "—";
  return `${value.toFixed(digits)}%`;
};

const formatDateTime = (iso, locale = "ru-RU") => {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString(locale, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
};

const formatDateShort = (iso, locale = "ru-RU") => {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleDateString(locale, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
};

const getVerdictPalette = (verdict) => {
  if (verdict === "Покупай") {
    return {
      bg: "#052e16",
      border: "#16a34a",
      text: "#4ade80",
      icon: "arrow-up-right",
    };
  }
  if (verdict === "Осторожно") {
    return {
      bg: "#450a0a",
      border: "#f97373",
      text: "#f97373",
      icon: "trending-down",
    };
  }
  // Neutral / other
  return {
    bg: "#451a03",
    border: "#fbbf24",
    text: "#facc15",
    icon: "alert-circle",
  };
};

const getHorizonLabel = (horizon, lang) => {
  if (lang === "ru") {
    if (horizon === 1) return "T+1 (следующий день)";
    if (horizon === 3) return "T+3 (3 дня вперёд)";
    return "T+7 (неделя вперёд)";
  }
  if (lang === "tr") {
    if (horizon === 1) return "T+1 (ertesi gün)";
    if (horizon === 3) return "T+3 (3 gün sonrası)";
    return "T+7 (1 hafta sonrası)";
  }
  // en
  if (horizon === 1) return "T+1 (next day)";
  if (horizon === 3) return "T+3 (3 days ahead)";
  return "T+7 (1 week ahead)";
};

const getProbTitle = (horizon, lang) => {
  if (lang === "ru") {
    return `Вероятность роста T+${horizon}`;
  }
  if (lang === "tr") {
    return `T+${horizon} yükseliş olasılığı`;
  }
  return `T+${horizon} upside probability`;
};

// ===========================
// MAIN COMPONENT
// ===========================

export default function App() {
  const [activeTab, setActiveTab] = useState("signals");
  const [lang, setLang] = useState("ru");
  const [horizon, setHorizon] = useState(1);

  const tr = useCallback((key) => translate(lang, key), [lang]);

  // signals for all assets
  const [signals, setSignals] = useState([]);
  const [loadingSignals, setLoadingSignals] = useState(true);
  const [signalsError, setSignalsError] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // selected asset
  const [selectedSymbol, setSelectedSymbol] = useState(null);

  // detailed prediction + history for chart
  const [predDetail, setPredDetail] = useState(null);
  const [historyPoints, setHistoryPoints] = useState([]);
  const [loadingPred, setLoadingPred] = useState(false);

  // backtest metrics
  const [stats, setStats] = useState([]);
  const [loadingStats, setLoadingStats] = useState(false);
  const [statsError, setStatsError] = useState(false);
  const [expandedSymbol, setExpandedSymbol] = useState(null);

  // ---- load signals (/signals) ----
  const loadSignals = useCallback(async () => {
    try {
      setLoadingSignals(true);
      setSignalsError(false);
      const resp = await axios.get(`${API_BASE_URL}/signals`, {
        params: { horizon_days: horizon },
      });
      const data = resp.data || [];
      setSignals(data);

      // if asset is not selected yet or doesn't exist for new horizon — select first in list
      setSelectedSymbol((prev) => {
        if (!data.length) return null;
        if (!prev) return data[0].symbol;
        const exists = data.some((x) => x.symbol === prev);
        return exists ? prev : data[0].symbol;
      });
    } catch (e) {
      console.error("loadSignals error:", e?.message || e);
      setSignalsError(true);
    } finally {
      setLoadingSignals(false);
    }
  }, [horizon]);

  useEffect(() => {
    loadSignals();
  }, [loadSignals]);

  // pull-to-refresh on Signals
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await loadSignals();
    setRefreshing(false);
  }, [loadSignals]);

  // ---- load prediction detail + history for selected asset & horizon ----
  useEffect(() => {
    const fetchPredAndHistory = async () => {
      if (!selectedSymbol) return;

      setLoadingPred(true);

      try {
        let predData = null;

        // 1. Онлайн-инференс для текущего горизонта
        try {
          const predResp = await axios.get(
            `${API_BASE_URL}/online_prediction/${selectedSymbol}`,
            { params: { horizon_days: horizon } }
          );
          predData = predResp.data;
        } catch (onlineErr) {
          console.error(
            "online_prediction error:",
            onlineErr?.message || onlineErr
          );

          // 2. Fallback: последняя оффлайн-запись
          try {
            const fallbackResp = await axios.get(
              `${API_BASE_URL}/prediction/${selectedSymbol}`,
              { params: { horizon_days: horizon } }
            );
            predData = fallbackResp.data;
          } catch (dbErr) {
            console.error(
              "fallback /prediction error:",
              dbErr?.message || dbErr
            );
          }
        }

        if (predData) {
          setPredDetail(predData);
        } else {
          setPredDetail(null);
        }

        // 3. History from DB — for chart
        try {
          const histResp = await axios.get(
            `${API_BASE_URL}/prediction/${selectedSymbol}/history`,
            { params: { limit: 30, horizon_days: horizon } }
          );
          setHistoryPoints(histResp.data || []);
        } catch (histErr) {
          console.error("history error:", histErr?.message || histErr);
          setHistoryPoints([]);
        }

        // 4. Refresh signals cards
        await loadSignals();
      } finally {
        setLoadingPred(false);
      }
    };

    fetchPredAndHistory();
  }, [selectedSymbol, loadSignals, horizon]);

  // ---- load backtest stats for current horizon ----
  const loadStats = useCallback(async () => {
    try {
      setLoadingStats(true);
      setStatsError(false);
      const resp = await axios.get(`${API_BASE_URL}/stats`, {
        params: { horizon_days: horizon },
      });
      setStats(resp.data || []);
    } catch (e) {
      console.error("loadStats error:", e?.message || e);
      setStatsError(true);
    } finally {
      setLoadingStats(false);
    }
  }, [horizon]);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  // ===========================
  // DERIVED DATA
  // ===========================

  const currentSignal = useMemo(
    () => signals.find((s) => s.symbol === selectedSymbol),
    [signals, selectedSymbol]
  );

  const upProb =
    predDetail?.up_prob ?? currentSignal?.up_prob ?? 0.5;

  const verdict =
    predDetail?.ux_verdict ?? currentSignal?.verdict ?? "Нейтрально";

  const predDate =
    predDetail?.pred_date ?? currentSignal?.last_update ?? null;

  const assetClass =
    selectedSymbol && ASSET_CLASS_CODE[selectedSymbol]
      ? tr(`asset.${ASSET_CLASS_CODE[selectedSymbol]}`)
      : tr("asset.default");

  const palette = getVerdictPalette(verdict);

  const confidenceCode = computeConfidenceCode(upProb);
  const confidenceLabel = tr(`confidence.${confidenceCode}`);

  const locale =
    lang === "tr" ? "tr-TR" : lang === "en" ? "en-US" : "ru-RU";

  const verdictLabel = VERDICT_LABELS[lang][verdict] ?? verdict;

  const horizonLabel = useMemo(
    () => getHorizonLabel(horizon, lang),
    [horizon, lang]
  );

  const probTitle = useMemo(
    () => getProbTitle(horizon, lang),
    [horizon, lang]
  );

  const historyData = useMemo(() => {
    if (!historyPoints || historyPoints.length === 0) return [];
    return historyPoints.map((p) => ({
      x: new Date(p.asof_time || p.pred_date),
      y: p.up_prob,
    }));
  }, [historyPoints]);

  // path for line and filled area
  const { linePath, areaPath } = useMemo(() => {
    if (!historyData || historyData.length < 2) {
      return { linePath: "", areaPath: "" };
    }

    const minY = Math.min(...historyData.map((p) => p.y));
    const maxY = Math.max(...historyData.map((p) => p.y));
    const rangeY = maxY - minY || 1;
    const stepX = CHART_WIDTH / (historyData.length - 1);

    let line = "";
    let area = `M 0 ${CHART_HEIGHT}`;

    historyData.forEach((point, idx) => {
      const x = idx * stepX;
      const norm = (point.y - minY) / rangeY; // 0..1
      const y =
        CHART_HEIGHT - norm * (CHART_HEIGHT - 20); // top padding

      if (idx === 0) {
        line = `M ${x} ${y}`;
      } else {
        line += ` L ${x} ${y}`;
      }
      area += ` L ${x} ${y}`;
    });

    const lastX = (historyData.length - 1) * stepX;
    area += ` L ${lastX} ${CHART_HEIGHT} Z`;

    return { linePath: line, areaPath: area };
  }, [historyData]);

  const buySignalsCount = useMemo(
    () =>
      signals.filter(
        (s) =>
          s.verdict === "Покупай" && (s.up_prob ?? 0) >= 0.6
      ).length,
    [signals]
  );

  const avgUpProb = useMemo(() => {
    if (!signals.length) return null;
    const s = signals.reduce((acc, x) => acc + (x.up_prob ?? 0), 0);
    return s / signals.length;
  }, [signals]);

  // ===========================
  // HELPERS: BACKTEST VIEW
  // ===========================

  const getModelVsBHTag = (st) => {
    const strat = st.strategy_return;
    const bh = st.buy_hold_return;
    if (typeof strat !== "number" || typeof bh !== "number") {
      return { label: tr("backtest.tag.mixed"), tone: "neutral" };
    }
    const diff = strat - bh;
    if (diff > 0.05) {
      return { label: tr("backtest.tag.modelBeatsBH"), tone: "good" };
    }
    if (diff < -0.05) {
      return { label: tr("backtest.tag.bhBeatsModel"), tone: "bad" };
    }
    return { label: tr("backtest.tag.mixed"), tone: "neutral" };
  };

  // ===========================
  // RENDER
  // ===========================

  return (
    <LinearGradient
      colors={["#020617", "#020617"]}
      style={{ flex: 1 }}
    >
      <SafeAreaView style={styles.container}>
        {/* HEADER */}
        <View style={styles.header}>
          <View>
            <Text style={styles.appTitle}>AI Signals</Text>
            <Text style={styles.appSubtitle}>
              {tr("header.subtitle")}
            </Text>
          </View>
          <View style={styles.headerRight}>
            <View style={styles.langSwitch}>
              {["ru", "tr", "en"].map((lng) => (
                <Pressable
                  key={lng}
                  onPress={() => setLang(lng)}
                  style={[
                    styles.langChip,
                    lang === lng && styles.langChipActive,
                  ]}
                >
                  <Text
                    style={[
                      styles.langChipText,
                      lang === lng && styles.langChipTextActive,
                    ]}
                  >
                    {lng.toUpperCase()}
                  </Text>
                </Pressable>
              ))}
            </View>
            <View style={styles.onlineBadge}>
              <Feather name="activity" size={14} color="#4ade80" />
              <Text style={styles.onlineText}>
                {tr("status.online")}
              </Text>
            </View>
          </View>
        </View>

        {/* TABS */}
        <View style={styles.tabsWrapper}>
          <View style={styles.tabs}>
            {["signals", "backtest", "about"].map((tab) => (
              <Pressable
                key={tab}
                onPress={() => setActiveTab(tab)}
                style={[
                  styles.tabButton,
                  activeTab === tab && styles.tabButtonActive,
                ]}
              >
                <Text
                  style={[
                    styles.tabText,
                    activeTab === tab && styles.tabTextActive,
                  ]}
                >
                  {tab === "signals"
                    ? tr("tabs.signals")
                    : tab === "backtest"
                    ? tr("tabs.backtest")
                    : tr("tabs.about")}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        {/* HORIZON SWITCH */}
        <View style={styles.horizonWrapper}>
          <View style={styles.horizonSwitch}>
            {[1, 3, 7].map((h) => (
              <Pressable
                key={h}
                onPress={() => setHorizon(h)}
                style={[
                  styles.horizonChip,
                  horizon === h && styles.horizonChipActive,
                ]}
              >
                <Text
                  style={[
                    styles.horizonChipText,
                    horizon === h && styles.horizonChipTextActive,
                  ]}
                >
                  {`T+${h}`}
                </Text>
              </Pressable>
            ))}
          </View>
        </View>

        {/* ========= TAB: SIGNALS ========= */}
        {activeTab === "signals" && (
          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={styles.scrollContent}
            refreshControl={
              <RefreshControl
                refreshing={refreshing}
                onRefresh={onRefresh}
                tintColor="#60a5fa"
              />
            }
          >
            {loadingSignals && (
              <View style={styles.center}>
                <ActivityIndicator size="large" color="#60a5fa" />
                <Text style={styles.loadingText}>
                  {tr("signals.loading")}
                </Text>
              </View>
            )}

            {signalsError && !loadingSignals && (
              <View style={styles.center}>
                <Text style={styles.errorText}>
                  {tr("signals.error")}
                </Text>
              </View>
            )}

            {!loadingSignals &&
              !signalsError &&
              signals.length > 0 && (
                <>
                  {/* ROW WITH ASSETS */}
                  <View style={styles.assetRow}>
                    {signals.map((s) => {
                      const isActive = s.symbol === selectedSymbol;
                      const change = s.change_24h;
                      const price = s.price;
                      return (
                        <Pressable
                          key={s.symbol}
                          onPress={() =>
                            setSelectedSymbol(s.symbol)
                          }
                          style={[
                            styles.assetCard,
                            isActive && styles.assetCardActive,
                          ]}
                        >
                          <View style={styles.assetCardHeader}>
                            <Text style={styles.assetSymbol}>
                              {s.symbol}
                            </Text>
                            {typeof change === "number" && (
                              <Text
                                style={[
                                  styles.assetChange,
                                  change >= 0
                                    ? styles.assetChangeUp
                                    : styles.assetChangeDown,
                                ]}
                              >
                                {change >= 0 ? "+" : ""}
                                {change.toFixed(2)}%
                              </Text>
                            )}
                          </View>
                          <Text style={styles.assetPrice}>
                            {formatUsd(price)}
                          </Text>
                        </Pressable>
                      );
                    })}
                  </View>

                  {/* MAIN CARD FOR SELECTED ASSET */}
                  {currentSignal && (
                    <View style={styles.mainCard}>
                      <View style={styles.mainCardHeader}>
                        <View>
                          <Text style={styles.mainSymbol}>
                            {selectedSymbol}
                          </Text>
                          <View style={styles.mainDateRow}>
                            <Feather
                              name="clock"
                              size={14}
                              color="#9ca3af"
                            />
                            <Text style={styles.mainDateText}>
                              {formatDateTime(predDate, locale)}
                            </Text>
                          </View>
                          <Text style={styles.assetClassText}>
                            {tr("asset.classLabel")} {assetClass}
                          </Text>
                        </View>

                        <View
                          style={[
                            styles.verdictBadge,
                            {
                              backgroundColor: palette.bg,
                              borderColor: palette.border,
                            },
                          ]}
                        >
                          <Feather
                            name={palette.icon}
                            size={16}
                            color={palette.text}
                          />
                          <Text
                            style={[
                              styles.verdictText,
                              { color: palette.text },
                            ]}
                          >
                            {verdictLabel}
                          </Text>
                        </View>
                      </View>

                      {/* up_prob chart from HISTORY */}
                      <View style={styles.chartWrapper}>
                        {linePath ? (
                          <Svg
                            width={CHART_WIDTH}
                            height={CHART_HEIGHT}
                          >
                            <Defs>
                              <SvgGradient
                                id="probGrad"
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                              >
                                <Stop
                                  offset="0"
                                  stopColor="#60a5fa"
                                  stopOpacity="0.6"
                                />
                                <Stop
                                  offset="1"
                                  stopColor="#60a5fa"
                                  stopOpacity="0"
                                />
                              </SvgGradient>
                            </Defs>
                            <Path
                              d={areaPath}
                              fill="url(#probGrad)"
                            />
                            <Path
                              d={linePath}
                              stroke="#60a5fa"
                              strokeWidth={2}
                              fill="none"
                              strokeLinecap="round"
                              strokeLinejoin="round"
                            />
                          </Svg>
                        ) : (
                          <View style={styles.chartEmpty}>
                            <Text style={styles.chartEmptyText}>
                              {tr("chart.noData")}
                            </Text>
                          </View>
                        )}
                      </View>

                      {/* T+N FORECAST */}
                      <View style={styles.probBlock}>
                        <Text style={styles.probLabel}>
                          {probTitle}
                        </Text>
                        <Text style={styles.probValue}>
                          {(upProb * 100).toFixed(1)}%
                        </Text>
                        <View style={styles.probBar}>
                          <View
                            style={[
                              styles.probBarFill,
                              {
                                width: `${Math.min(
                                  100,
                                  Math.max(0, upProb * 100)
                                )}%`,
                              },
                            ]}
                          />
                        </View>
                      </View>

                      {/* SHORT PORTFOLIO AGGREGATES */}
                      <View style={styles.inlineStatsRow}>
                        <View style={styles.inlineStatCol}>
                          <Text style={styles.inlineLabel}>
                            {tr("inline.buyCount")}
                          </Text>
                          <Text style={styles.inlineValue}>
                            {buySignalsCount}
                          </Text>
                        </View>
                        <View style={styles.inlineStatCol}>
                          <Text style={styles.inlineLabel}>
                            {tr("inline.avgUpProb")}
                          </Text>
                          <Text style={styles.inlineValue}>
                            {avgUpProb != null
                              ? `${(avgUpProb * 100).toFixed(1)}%`
                              : "—"}
                          </Text>
                        </View>
                      </View>

                      {/* METRICS GRID */}
                      <View style={styles.metricsGrid}>
                        <View style={styles.metricCard}>
                          <View style={styles.metricHeader}>
                            <Feather
                              name="target"
                              size={16}
                              color="#60a5fa"
                            />
                            <Text style={styles.metricLabel}>
                              {tr("metrics.probability")}
                            </Text>
                          </View>
                          <Text style={styles.metricValueBlue}>
                            {(upProb * 100).toFixed(0)}%
                          </Text>
                        </View>

                        <View style={styles.metricCard}>
                          <View style={styles.metricHeader}>
                            <Feather
                              name="activity"
                              size={16}
                              color="#f59e0b"
                            />
                            <Text style={styles.metricLabel}>
                              {tr("metrics.confidence")}
                            </Text>
                          </View>
                          <Text style={styles.metricValueYellow}>
                            {confidenceLabel}
                          </Text>
                        </View>

                        <View style={styles.metricCard}>
                          <View style={styles.metricHeader}>
                            <Feather
                              name="calendar"
                              size={16}
                              color="#a855f7"
                            />
                            <Text style={styles.metricLabel}>
                              {tr("metrics.predDate")}
                            </Text>
                          </View>
                          <Text style={styles.metricValuePurple}>
                            {formatDateShort(predDate, locale)}
                          </Text>
                        </View>

                        <View style={styles.metricCard}>
                          <View style={styles.metricHeader}>
                            <Feather
                              name="info"
                              size={16}
                              color="#38bdf8"
                            />
                            <Text style={styles.metricLabel}>
                              {tr("metrics.horizon")}
                            </Text>
                          </View>
                          <Text style={styles.metricValuePurple}>
                            {horizonLabel}
                          </Text>
                        </View>
                      </View>

                      {/* ACTION BUTTON */}
                      <Pressable
                        style={[
                          styles.actionButton,
                          verdict === "Покупай"
                            ? styles.actionButtonBuy
                            : verdict === "Осторожно"
                            ? styles.actionButtonSell
                            : styles.actionButtonNeutral,
                        ]}
                      >
                        <Text style={styles.actionButtonText}>
                          {verdict === "Покупай"
                            ? tr("scenario.buy")
                            : verdict === "Осторожно"
                            ? tr("scenario.caution")
                            : tr("scenario.neutral")}
                        </Text>
                        <Feather
                          name="chevron-right"
                          size={18}
                          color="#ffffff"
                        />
                      </Pressable>
                    </View>
                  )}

                  {/* LIST OF ALL SIGNALS */}
                  <View style={styles.allSignalsCard}>
                    <View style={styles.allSignalsHeader}>
                      <Text style={styles.allSignalsTitle}>
                        {tr("list.allSignals")}
                      </Text>
                    </View>

                    {signals.map((s) => {
                      const paletteRow = getVerdictPalette(s.verdict);
                      const change = s.change_24h;
                      const price = s.price;
                      const rowVerdictLabel =
                        VERDICT_LABELS[lang][s.verdict] ??
                        s.verdict;

                      return (
                        <Pressable
                          key={s.symbol}
                          onPress={() =>
                            setSelectedSymbol(s.symbol)
                          }
                          style={styles.signalRow}
                        >
                          <View style={styles.signalLeft}>
                            <LinearGradient
                              colors={["#4f46e5", "#a855f7"]}
                              style={styles.assetIcon}
                            >
                              <Text style={styles.assetIconText}>
                                {s.symbol
                                  .slice(0, 2)
                                  .toUpperCase()}
                              </Text>
                            </LinearGradient>
                            <View>
                              <Text style={styles.signalSymbol}>
                                {s.symbol}
                              </Text>
                              <Text style={styles.signalPrice}>
                                {formatUsd(price)}
                              </Text>
                            </View>
                          </View>

                          <View style={styles.signalRight}>
                            <View style={styles.signalChangeBlock}>
                              {typeof change === "number" && (
                                <Text
                                  style={[
                                    styles.signalChange,
                                    change >= 0
                                      ? styles.assetChangeUp
                                      : styles.assetChangeDown,
                                  ]}
                                >
                                  {change >= 0 ? "+" : ""}
                                  {change.toFixed(2)}%
                                </Text>
                              )}
                              <Text
                                style={styles.signalChangeLabel}
                              >
                                {tr("list.change24h")}
                              </Text>
                            </View>

                            <View
                              style={[
                                styles.signalVerdictChip,
                                {
                                  backgroundColor: paletteRow.bg,
                                  borderColor: paletteRow.border,
                                },
                              ]}
                            >
                              <Text
                                style={[
                                  styles.signalVerdictText,
                                  { color: paletteRow.text },
                                ]}
                              >
                                {rowVerdictLabel}
                              </Text>
                            </View>
                          </View>
                        </Pressable>
                      );
                    })}
                  </View>
                </>
              )}
          </ScrollView>
        )}

        {/* ========= TAB: BACKTEST ========= */}
        {activeTab === "backtest" && (
          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={styles.scrollContent}
          >
            <View style={styles.portfolioCard}>
              <Text style={styles.cardTitle}>
                {tr("backtest.title")}
              </Text>
              <Text style={styles.aboutTextSmall}>
                {tr("backtest.subtitle")}
              </Text>
            </View>

            {loadingStats && (
              <View style={styles.center}>
                <ActivityIndicator size="large" color="#60a5fa" />
                <Text style={styles.loadingText}>
                  {tr("backtest.loading")}
                </Text>
              </View>
            )}

            {statsError && !loadingStats && (
              <View style={styles.center}>
                <Text style={styles.errorText}>
                  {tr("backtest.error")}
                </Text>
              </View>
            )}

            {!loadingStats &&
              !statsError &&
              stats.length > 0 && (
                <View style={styles.modelGrid}>
                  {stats.map((st) => {
                    const isExpanded =
                      expandedSymbol === st.symbol;
                    const tag = getModelVsBHTag(st);

                    return (
                      <Pressable
                        key={st.symbol}
                        onPress={() =>
                          setExpandedSymbol((prev) =>
                            prev === st.symbol ? null : st.symbol
                          )
                        }
                        style={[
                          styles.modelCard,
                          isExpanded && styles.modelCardExpanded,
                        ]}
                      >
                        <View style={styles.modelHeaderRow}>
                          <Text style={styles.modelSymbol}>
                            {st.symbol}
                          </Text>

                          <View
                            style={[
                              styles.modelTag,
                              tag.tone === "good" &&
                                styles.modelTagGood,
                              tag.tone === "bad" &&
                                styles.modelTagBad,
                            ]}
                          >
                            <Text style={styles.modelTagText}>
                              {tag.label}
                            </Text>
                          </View>
                        </View>

                        {/* Classification quality */}
                        <View style={styles.modelSectionHeader}>
                          <Feather
                            name="target"
                            size={14}
                            color="#60a5fa"
                          />
                          <Text
                            style={styles.modelSectionTitle}
                          >
                            {tr("metrics.probability")}
                          </Text>
                        </View>

                        <View style={styles.modelMetricRow}>
                          <Text style={styles.modelMetricLabel}>
                            {tr("backtest.metric.balAcc")}
                          </Text>
                          <Text style={styles.modelMetricValue}>
                            {st.bal_acc != null
                              ? formatPercent(
                                  st.bal_acc * 100,
                                  1
                                )
                              : "—"}
                          </Text>
                        </View>
                        <View style={styles.modelMetricRow}>
                          <Text style={styles.modelMetricLabel}>
                            {tr("backtest.metric.auc")}
                          </Text>
                          <Text style={styles.modelMetricValue}>
                            {st.auc != null
                              ? st.auc.toFixed(3)
                              : "—"}
                          </Text>
                        </View>
                        {typeof st.pred_pos_rate === "number" && (
                          <View style={styles.modelMetricRow}>
                            <Text
                              style={styles.modelMetricLabel}
                            >
                              {tr("backtest.metric.posRate")}
                            </Text>
                            <Text
                              style={styles.modelMetricValue}
                            >
                              {formatPercent(
                                st.pred_pos_rate * 100,
                                1
                              )}
                            </Text>
                          </View>
                        )}

                        {/* Strategy vs Buy&Hold */}
                        <View
                          style={[
                            styles.modelSectionHeader,
                            { marginTop: 8 },
                          ]}
                        >
                          <Feather
                            name="activity"
                            size={14}
                            color="#22c55e"
                          />
                          <Text
                            style={styles.modelSectionTitle}
                          >
                            {tr(
                              "backtest.metric.strategyReturn"
                            )}
                          </Text>
                        </View>

                        <View style={styles.modelMetricRow}>
                          <Text style={styles.modelMetricLabel}>
                            {tr(
                              "backtest.metric.strategyReturn"
                            )}
                          </Text>
                          <Text
                            style={[
                              styles.modelMetricValue,
                              st.strategy_return != null &&
                              st.strategy_return < 0
                                ? styles.modelMetricValueRed
                                : styles.modelMetricValueGreen,
                            ]}
                          >
                            {st.strategy_return != null
                              ? formatPercent(
                                  st.strategy_return,
                                  2
                                )
                              : "—"}
                          </Text>
                        </View>
                        <View style={styles.modelMetricRow}>
                          <Text style={styles.modelMetricLabel}>
                            {tr(
                              "backtest.metric.buyHoldReturn"
                            )}
                          </Text>
                          <Text style={styles.modelMetricValue}>
                            {st.buy_hold_return != null
                              ? formatPercent(
                                  st.buy_hold_return,
                                  2
                                )
                              : "—"}
                          </Text>
                        </View>
                        {typeof st.excess_return === "number" && (
                          <View style={styles.modelMetricRow}>
                            <Text
                              style={styles.modelMetricLabel}
                            >
                              {tr(
                                "backtest.metric.excessReturn"
                              )}
                            </Text>
                            <Text
                              style={[
                                styles.modelMetricValue,
                                st.excess_return < 0
                                  ? styles.modelMetricValueRed
                                  : styles.modelMetricValueGreen,
                              ]}
                            >
                              {formatPercent(
                                st.excess_return,
                                2
                              )}
                            </Text>
                          </View>
                        )}

                        {/* Collapsed/expanded extra metrics */}
                        {isExpanded && (
                          <>
                            <View
                              style={[
                                styles.modelSectionHeader,
                                { marginTop: 8 },
                              ]}
                            >
                              <Feather
                                name="bar-chart-2"
                                size={14}
                                color="#a855f7"
                              />
                              <Text
                                style={
                                  styles.modelSectionTitle
                                }
                              >
                                Risk & trades
                              </Text>
                            </View>

                            <View style={styles.modelMetricRow}>
                              <Text
                                style={styles.modelMetricLabel}
                              >
                                {tr("backtest.metric.sharpe")}
                              </Text>
                              <Text
                                style={styles.modelMetricValue}
                              >
                                {typeof st.sharpe === "number"
                                  ? st.sharpe.toFixed(2)
                                  : "—"}
                              </Text>
                            </View>
                            {typeof st.sharpe_conf ===
                              "number" && (
                              <View
                                style={styles.modelMetricRow}
                              >
                                <Text
                                  style={
                                    styles.modelMetricLabel
                                  }
                                >
                                  {tr(
                                    "backtest.metric.sharpeConf"
                                  )}
                                </Text>
                                <Text
                                  style={
                                    styles.modelMetricValue
                                  }
                                >
                                  {st.sharpe_conf.toFixed(2)}
                                </Text>
                              </View>
                            )}

                            <View style={styles.modelMetricRow}>
                              <Text
                                style={styles.modelMetricLabel}
                              >
                                {tr("backtest.metric.winRate")}
                              </Text>
                              <Text
                                style={styles.modelMetricValue}
                              >
                                {typeof st.win_rate ===
                                "number"
                                  ? formatPercent(
                                      st.win_rate * 100,
                                      1
                                    )
                                  : "—"}
                              </Text>
                            </View>
                            {typeof st.win_rate_conf ===
                              "number" && (
                              <View
                                style={styles.modelMetricRow}
                              >
                                <Text
                                  style={
                                    styles.modelMetricLabel
                                  }
                                >
                                  {tr(
                                    "backtest.metric.winRateConf"
                                  )}
                                </Text>
                                <Text
                                  style={
                                    styles.modelMetricValue
                                  }
                                >
                                  {formatPercent(
                                    st.win_rate_conf * 100,
                                    1
                                  )}
                                </Text>
                              </View>
                            )}

                            {typeof st.coverage ===
                              "number" && (
                              <View
                                style={styles.modelMetricRow}
                              >
                                <Text
                                  style={
                                    styles.modelMetricLabel
                                  }
                                >
                                  {tr(
                                    "backtest.metric.coverage"
                                  )}
                                </Text>
                                <Text
                                  style={
                                    styles.modelMetricValue
                                  }
                                >
                                  {formatPercent(
                                    st.coverage * 100,
                                    1
                                  )}
                                </Text>
                              </View>
                            )}

                            {typeof st.executed_trades_all ===
                              "number" && (
                              <View
                                style={styles.modelMetricRow}
                              >
                                <Text
                                  style={
                                    styles.modelMetricLabel
                                  }
                                >
                                  {tr(
                                    "backtest.metric.tradesAll"
                                  )}
                                </Text>
                                <Text
                                  style={
                                    styles.modelMetricValue
                                  }
                                >
                                  {st.executed_trades_all}
                                </Text>
                              </View>
                            )}
                            {typeof st.executed_trades_conf ===
                              "number" && (
                              <View
                                style={styles.modelMetricRow}
                              >
                                <Text
                                  style={
                                    styles.modelMetricLabel
                                  }
                                >
                                  {tr(
                                    "backtest.metric.tradesConf"
                                  )}
                                </Text>
                                <Text
                                  style={
                                    styles.modelMetricValue
                                  }
                                >
                                  {st.executed_trades_conf}
                                </Text>
                              </View>
                            )}
                          </>
                        )}

                        {/* Test period */}
                        <View
                          style={[
                            styles.modelMetricRow,
                            { marginTop: 6 },
                          ]}
                        >
                          <Text style={styles.modelMetricLabel}>
                            {tr("backtest.metric.testPeriod")}
                          </Text>
                          <Text style={styles.modelMetricValue}>
                            {formatDateShort(
                              st.test_period_start,
                              locale
                            )}{" "}
                            –{" "}
                            {formatDateShort(
                              st.test_period_end,
                              locale
                            )}
                          </Text>
                        </View>

                        <View style={styles.modelExpandHint}>
                          <Feather
                            name={
                              isExpanded
                                ? "chevron-up"
                                : "chevron-down"
                            }
                            size={14}
                            color="#6b7280"
                          />
                          <Text
                            style={styles.modelExpandHintText}
                          >
                            {isExpanded
                              ? "Hide details"
                              : "Show details"}
                          </Text>
                        </View>
                      </Pressable>
                    );
                  })}
                </View>
              )}
          </ScrollView>
        )}

        {/* ========= TAB: ABOUT ========= */}
        {activeTab === "about" && (
          <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={styles.scrollContent}
          >
            {/* High-level intro */}
            <View style={styles.portfolioCard}>
              <Text style={styles.cardTitle}>
                {tr("about.title")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.main")}
              </Text>
            </View>

            {/* Assets */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.assetsTitle")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.assetsText")}
              </Text>
            </View>

            {/* Horizons */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.t1Title")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.t1Text")}
              </Text>
            </View>

            {/* How to read signals */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.readSignalsTitle")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.readSignalsText")}
              </Text>
            </View>

            {/* Training setup */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.trainingTitle")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.trainingText")}
              </Text>
            </View>

            {/* How to use in practice */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.howToUseTitle")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.howToUseText")}
              </Text>
            </View>

            {/* Pipeline */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.pipelineTitle")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.pipelineText")}
              </Text>
            </View>

            {/* Tech stack */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.techTitle")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.techText")}
              </Text>
            </View>

            {/* Limitations / disclaimer */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.limitationsTitle")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.limitationsText")}
              </Text>
            </View>

            {/* FAQ */}
            <View style={styles.portfolioCard}>
              <Text style={styles.aboutSectionTitle}>
                {tr("about.faqTitle")}
              </Text>
              <Text style={styles.aboutText}>
                {tr("about.faqText")}
              </Text>
            </View>
          </ScrollView>
        )}

        {/* Loader for updating prediction for selected asset */}
        {loadingPred && activeTab === "signals" && (
          <View pointerEvents="none" style={styles.predLoading}>
            <ActivityIndicator size="small" color="#60a5fa" />
            <Text style={styles.predLoadingText}>
              {tr("pred.loading")} {selectedSymbol}...
            </Text>
          </View>
        )}
      </SafeAreaView>
    </LinearGradient>
  );
}

// ===========================
// STYLES
// ===========================

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "transparent",
  },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  headerRight: {
    flexDirection: "row",
    alignItems: "center",
  },
  langSwitch: {
    flexDirection: "row",
    padding: 2,
    borderRadius: 999,
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#1f2937",
    marginRight: 8,
  },
  langChip: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 999,
  },
  langChipActive: {
    backgroundColor: "#1d4ed8",
  },
  langChipText: {
    fontSize: 11,
    color: "#9ca3af",
    fontWeight: "600",
  },
  langChipTextActive: {
    color: "#ffffff",
  },
  appTitle: {
    fontSize: 22,
    fontWeight: "800",
    color: "#38bdf8",
  },
  appSubtitle: {
    fontSize: 13,
    color: "#9ca3af",
    marginTop: 2,
  },
  onlineBadge: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: "rgba(34,197,94,0.08)",
    borderWidth: 1,
    borderColor: "rgba(34,197,94,0.6)",
  },
  onlineText: {
    marginLeft: 6,
    color: "#4ade80",
    fontSize: 12,
    fontWeight: "600",
  },
  tabsWrapper: {
    paddingHorizontal: 16,
    paddingBottom: 8,
  },
  tabs: {
    flexDirection: "row",
    backgroundColor: "#020617",
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#1f2937",
    padding: 4,
  },
  tabButton: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 999,
    alignItems: "center",
    justifyContent: "center",
  },
  tabButtonActive: {
    backgroundColor: "#2563eb",
  },
  tabText: {
    fontSize: 13,
    color: "#9ca3af",
    fontWeight: "500",
  },
  tabTextActive: {
    color: "#ffffff",
    fontWeight: "600",
  },
  horizonWrapper: {
    paddingHorizontal: 16,
    paddingBottom: 8,
  },
  horizonSwitch: {
    flexDirection: "row",
    alignSelf: "flex-start",
    backgroundColor: "#020617",
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#1f2937",
    padding: 3,
  },
  horizonChip: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    marginHorizontal: 2,
  },
  horizonChipActive: {
    backgroundColor: "#0f172a",
    borderWidth: 1,
    borderColor: "#60a5fa",
  },
  horizonChipText: {
    fontSize: 11,
    color: "#9ca3af",
    fontWeight: "600",
  },
  horizonChipTextActive: {
    color: "#e5e7eb",
  },
  scrollContent: {
    paddingHorizontal: 16,
    paddingBottom: 24,
  },
  center: {
    flex: 1,
    alignItems: "center",
    justifyContent: "center",
    paddingHorizontal: 24,
  },
  loadingText: {
    marginTop: 8,
    color: "#9ca3af",
  },
  errorText: {
    color: "#f97373",
    textAlign: "center",
  },
  assetRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 16,
    marginBottom: 16,
  },
  assetCard: {
    flex: 1,
    marginHorizontal: 4,
    backgroundColor: "#020617",
    borderRadius: 14,
    paddingVertical: 10,
    paddingHorizontal: 10,
    borderWidth: 1,
    borderColor: "#111827",
  },
  assetCardActive: {
    backgroundColor: "#1d4ed8",
    borderColor: "#60a5fa",
    shadowColor: "#1d4ed8",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.4,
    shadowRadius: 14,
    elevation: 5,
  },
  assetCardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 6,
  },
  assetSymbol: {
    color: "#ffffff",
    fontSize: 15,
    fontWeight: "700",
  },
  assetChange: {
    fontSize: 11,
    fontWeight: "600",
  },
  assetChangeUp: {
    color: "#4ade80",
  },
  assetChangeDown: {
    color: "#f97373",
  },
  assetPrice: {
    color: "#9ca3af",
    fontSize: 11,
  },
  mainCard: {
    backgroundColor: "#020617",
    borderRadius: 18,
    padding: 16,
    borderWidth: 1,
    borderColor: "#111827",
    marginBottom: 18,
  },
  mainCardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 10,
  },
  mainSymbol: {
    color: "#ffffff",
    fontSize: 24,
    fontWeight: "800",
  },
  mainDateRow: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 4,
  },
  mainDateText: {
    marginLeft: 6,
    color: "#9ca3af",
    fontSize: 12,
  },
  assetClassText: {
    marginTop: 2,
    color: "#6b7280",
    fontSize: 11,
  },
  verdictBadge: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    borderWidth: 1,
  },
  verdictText: {
    marginLeft: 6,
    fontSize: 13,
    fontWeight: "700",
  },
  chartWrapper: {
    marginTop: 12,
    backgroundColor: "#020617",
    alignItems: "center",
    justifyContent: "center",
  },
  chartEmpty: {
    height: CHART_HEIGHT,
    alignItems: "center",
    justifyContent: "center",
  },
  chartEmptyText: {
    color: "#6b7280",
    fontSize: 12,
  },
  probBlock: {
    marginTop: 4,
  },
  probLabel: {
    color: "#9ca3af",
    fontSize: 12,
  },
  probValue: {
    color: "#facc15",
    fontSize: 22,
    fontWeight: "800",
    marginTop: 2,
  },
  probBar: {
    marginTop: 6,
    height: 6,
    borderRadius: 999,
    backgroundColor: "#111827",
    overflow: "hidden",
  },
  probBarFill: {
    height: "100%",
    backgroundColor: "#a3e635",
  },
  inlineStatsRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 12,
  },
  inlineStatCol: {
    flex: 1,
    marginRight: 8,
  },
  inlineLabel: {
    color: "#6b7280",
    fontSize: 11,
  },
  inlineValue: {
    color: "#e5e7eb",
    fontSize: 15,
    fontWeight: "600",
    marginTop: 2,
  },
  metricsGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    marginTop: 12,
  },
  metricCard: {
    width: "50%",
    padding: 8,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#111827",
    backgroundColor: "#020617",
  },
  metricHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 4,
  },
  metricLabel: {
    marginLeft: 6,
    color: "#9ca3af",
    fontSize: 11,
  },
  metricValueBlue: {
    color: "#60a5fa",
    fontSize: 20,
    fontWeight: "800",
  },
  metricValuePurple: {
    color: "#c4b5fd",
    fontSize: 18,
    fontWeight: "700",
  },
  metricValueYellow: {
    color: "#facc15",
    fontSize: 18,
    fontWeight: "700",
  },
  actionButton: {
    marginTop: 14,
    paddingVertical: 12,
    borderRadius: 999,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
  },
  actionButtonBuy: {
    backgroundColor: "#22c55e",
  },
  actionButtonSell: {
    backgroundColor: "#ef4444",
  },
  actionButtonNeutral: {
    backgroundColor: "#374151",
  },
  actionButtonText: {
    color: "#ffffff",
    fontSize: 16,
    fontWeight: "700",
    marginRight: 6,
  },
  allSignalsCard: {
    backgroundColor: "#020617",
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "#111827",
    overflow: "hidden",
  },
  allSignalsHeader: {
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: "#111827",
  },
  allSignalsTitle: {
    color: "#e5e7eb",
    fontSize: 16,
    fontWeight: "700",
  },
  signalRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: "#020617",
  },
  signalLeft: {
    flexDirection: "row",
    alignItems: "center",
  },
  assetIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    marginRight: 10,
  },
  assetIconText: {
    color: "#ffffff",
    fontWeight: "800",
    fontSize: 14,
  },
  signalSymbol: {
    color: "#f9fafb",
    fontSize: 14,
    fontWeight: "600",
  },
  signalPrice: {
    color: "#9ca3af",
    fontSize: 12,
    marginTop: 2,
  },
  signalRight: {
    flexDirection: "row",
    alignItems: "center",
  },
  signalChangeBlock: {
    alignItems: "flex-end",
    marginRight: 12,
  },
  signalChange: {
    fontSize: 13,
    fontWeight: "600",
  },
  signalChangeLabel: {
    color: "#6b7280",
    fontSize: 10,
    marginTop: 2,
  },
  signalVerdictChip: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 999,
    borderWidth: 1,
  },
  signalVerdictText: {
    fontSize: 12,
    fontWeight: "600",
  },
  predLoading: {
    position: "absolute",
    bottom: 8,
    alignSelf: "center",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
    backgroundColor: "rgba(15,23,42,0.95)",
    flexDirection: "row",
    alignItems: "center",
  },
  predLoadingText: {
    marginLeft: 6,
    color: "#e5e7eb",
    fontSize: 11,
  },

  // === CARDS FOR BACKTEST / ABOUT ===
  portfolioCard: {
    backgroundColor: "#020617",
    borderRadius: 18,
    borderWidth: 1,
    borderColor: "#111827",
    padding: 16,
    marginBottom: 16,
  },
  cardTitle: {
    color: "#e5e7eb",
    fontSize: 16,
    fontWeight: "700",
  },
  aboutSectionTitle: {
    color: "#e5e7eb",
    fontSize: 15,
    fontWeight: "700",
    marginBottom: 8,
  },
  aboutText: {
    color: "#9ca3af",
    fontSize: 13,
    lineHeight: 18,
    marginTop: 6,
  },
  aboutTextSmall: {
    color: "#9ca3af",
    fontSize: 12,
    lineHeight: 17,
    marginTop: 6,
  },

  // === BACKTEST GRID ===
  modelGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    marginTop: 12,
  },
  modelCard: {
    width: "48%",
    padding: 12,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: "#111827",
    backgroundColor: "#020617",
    marginTop: 12,
  },
  modelCardExpanded: {
    borderColor: "#2563eb",
  },
  modelHeaderRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 4,
  },
  modelSymbol: {
    color: "#f9fafb",
    fontSize: 16,
    fontWeight: "700",
  },
  modelTag: {
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 999,
    borderWidth: 1,
    borderColor: "#374151",
    backgroundColor: "#020617",
  },
  modelTagGood: {
    borderColor: "#4ade80",
    backgroundColor: "rgba(34,197,94,0.08)",
  },
  modelTagBad: {
    borderColor: "#f97373",
    backgroundColor: "rgba(248,113,113,0.08)",
  },
  modelTagText: {
    color: "#e5e7eb",
    fontSize: 10,
    fontWeight: "600",
  },
  modelSectionHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 6,
    marginBottom: 2,
  },
  modelSectionTitle: {
    marginLeft: 6,
    color: "#9ca3af",
    fontSize: 11,
    fontWeight: "600",
    textTransform: "uppercase",
  },
  modelMetricRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 2,
  },
  modelMetricLabel: {
    color: "#9ca3af",
    fontSize: 11,
  },
  modelMetricValue: {
    color: "#e5e7eb",
    fontSize: 13,
    fontWeight: "600",
  },
  modelMetricValueGreen: {
    color: "#4ade80",
  },
  modelMetricValueRed: {
    color: "#f97373",
  },
  modelExpandHint: {
    marginTop: 6,
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "flex-end",
  },
  modelExpandHintText: {
    marginLeft: 4,
    color: "#6b7280",
    fontSize: 10,
  },
});
