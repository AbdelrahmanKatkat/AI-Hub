# Data Science — Curriculum (zero → hero)

Goal: a learner who knows basic algebra and can read a table ends able to take a messy real-world dataset through **questions**,
**cleaning**, **exploration**, **statistical reasoning**, **experiments**, **features**, **models**, **forecasts** and **communication**, and
to ship the result responsibly so that other people can trust and reuse it.

**Emphases:** (1) **The question first** — every lesson starts with a decision someone must make and the data that could inform it.
(2) **Animation** — rows move through pipelines, distributions reshape as samples grow, confidence intervals open and close, p-values
accumulate, and models are fitted and refitted as the learner changes data or settings. (3) **Honest numbers** — every example is
computed on a small dataset that is printed in the lesson, and every simulation is labelled *illustrative* when it stands in for a larger study.

Order is strict. Ten modules, 59 lessons. Module pages are `ds-<name>.html`; hub `data-science.html`.

Maths and ML prerequisites are linked, not re-taught: `math-probability.html` (distributions, expectation, Bayes), `math-statistics.html`
(estimation, likelihood), `math-linear-algebra.html` (matrices of features), `math-optimization.html` (loss minimisation), `ml-linear.html`
, `ml-classical.html` and `ml-evaluation.html` (models and metrics), `ml-neural.html` (neural models), `geo-foundations.html` (spatial data, shared with GeoAI).

## Lesson template (every lesson must have all of these)
1. **The question** – a concrete business, science or policy question that makes the idea necessary.
2. **Intuition** – plain words and a picture of the data first.
3. **Build it up** – step by step with a hand-worked example on a 5–10 row table or a tiny synthetic dataset (real numbers).
4. **Formal definition** – equations and a symbol table.
5. **Animated lab** – the learner changes a sample size, a threshold, a transformation or a model setting and sees the result move; "Try this" steps.
6. **Common traps**.
7. **Where it shows up** – real datasets, tools and products.
8. **Check yourself** – 3 questions with hidden answers.
9. **Takeaways**.

---
## Module 0 · Foundations of data science  (`ds-foundations.html`)
*Animation focus: a question flowing into a dataset, a table of rows being sorted and filtered, the data-science loop running around once.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| what-is-data-science | What data science is: questions, data, evidence, decisions | the data-science loop (question → data → analysis → decision → monitoring); descriptive, diagnostic, predictive and prescriptive questions | every ML project starts here |
| data-types | Types of data: tables, text, images, time, graphs | rows as observations and columns as variables; numeric, categorical, ordinal and datetime types; structured versus unstructured data | feature choice, model input |
| tidy-data | Tidy data and the unit of analysis | one observation per row; one variable per column; choosing the unit (customer, session, patient) | every analysis pipeline |
| tools-and-reproducibility | The working environment: notebooks, scripts and versions | scripted analysis over clicked analysis; seeds; environments and version pinning; a reproducible project folder | research and production code |
| asking-good-questions | Turning a vague wish into an answerable question | a measurable outcome; a comparison or baseline; the decision the answer will change; what would count as a failure | A/B tests, forecasting targets |

## Module 1 · Data wrangling & cleaning  (`ds-wrangling.html`)
*Animation focus: rows flowing through filter, join and group operations; a join matching keys between two tables; missing cells lighting up and being filled.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| selecting-filtering | Selecting, filtering and sorting rows and columns | boolean masks; row versus column selection; chaining operations; keeping the original file untouched | every data pipeline |
| joins | Joining tables: inner, left, outer and many-to-one | keys; duplicate keys that multiply rows; checking row counts before and after a join | combining user, event and label tables |
| missing-data | Missing data: types and handling | MCAR, MAR and MNAR; deletion versus imputation; indicator columns that record missingness | survey data, sensor logs, clinical records |
| duplicates-errors | Duplicates, outliers and impossible values | exact versus near duplicates; range checks; separating errors from rare true values | data quality in every production system |
| types-units | Parsing types, units and formats | string-to-number parsing; currencies and time zones; unit consistency; categorical codes | feature pipelines, logs |
| aggregation | Grouping and aggregating: split, apply, combine | group keys; sum, mean and count per group; the effect of unequal group sizes; pivot tables | dashboards, cohort analysis |

## Module 2 · Exploratory data analysis  (`ds-eda.html`)
*Animation focus: histograms and box plots redrawing as bins change, scatter plots gaining points, a correlation matrix filling in, and outliers highlighted when the threshold moves.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| univariate-summary | One variable: centre, spread and shape | mean, median and mode; variance and IQR; skew; the five-number summary | feature inspection before modelling |
| distributions-shape | Distribution shapes and transformations | histograms and bin width; log and Box–Cox transforms; heavy tails | income, counts, latency data |
| relationships | Two variables: scatter plots, correlation and its limits | Pearson and Spearman correlation; non-linear dependence that correlation misses; Anscombe's quartet | feature selection |
| categorical-eda | Categorical variables and cross-tabs | frequency tables; cross-tabulation; rare categories; Simpson's paradox as a warning | segment analysis, bias checks |
| groups-compare | Comparing groups and the effect of sample size | side-by-side box plots; effect size versus p-value (preview); how bins and groups can mislead | group comparisons, fairness audits |
| eda-workflow | An EDA workflow that ends in a decision | an EDA checklist; writing down what you looked at and what changed; deciding what to model | every project |

## Module 3 · Statistics for data science  (`ds-statistics.html`)
*Animation focus: repeated samples drawn from a population, the sample mean settling over time, the sampling distribution forming, confidence intervals opening and covering the true value.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| sampling | Populations, samples and sampling bias | random versus convenience samples; selection bias; sampling frames; why more data from the wrong source does not help | survey design, training-set bias |
| sampling-distribution | The sampling distribution and the central limit theorem | the sampling distribution of the mean; standard error σ/√n; CLT with a skewed population | every confidence interval |
| confidence-intervals | Confidence intervals and what they mean | interval construction x̄ ± z·SE; coverage; the common misreading of a 95% interval | reporting model metrics |
| hypothesis-testing | Hypothesis tests: null, alternative, test statistic and p-value | H0 and H1; one-sample t-test by hand; the p-value as P(data this extreme or more \| H0); significance level α | A/B test analysis |
| errors-power | Type I and II errors, power and sample size | α, β and power 1−β; the effect of effect size and n; choosing n before the experiment | experiment planning |
| bootstrap | The bootstrap: uncertainty without formulas | resampling with replacement; percentile intervals; when the bootstrap fails (heavy tails, tiny n) | model uncertainty, small datasets |

## Module 4 · Visualisation & storytelling  (`ds-visualisation.html`)
*Animation focus: a chart built one encoding at a time, the same data shown with a truncated and a full axis, and a dashboard refreshing as a filter changes.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| encodings | Visual encodings: position, length, colour and area | the ranking of encodings by accuracy of perception; avoiding area for quantities; pie charts and their limits | every chart |
| choosing-charts | Choosing the right chart for the question | comparison, composition, distribution, relationship and trend; line charts for time; bars for categories | reports and dashboards |
| honest-axes | Honest axes, scales and baselines | zero baselines for bars; log scales with care; dual axes that imply false relationships | communicating model results |
| colour-accessibility | Colour, contrast and colour-blind safe palettes | sequential, diverging and categorical palettes; contrast ratios; redundant encoding | public-facing analytics |
| dashboards | Dashboards: what to show and what to leave out | the audience and decision; a small number of KPIs; filters that change the story; drill-down | business intelligence |
| storytelling | Telling the story: from chart to argument | a headline that states the finding; the one chart that carries the argument; uncertainty in the picture | stakeholder reports |

## Module 5 · Experiments & causal inference  (`ds-experiments.html`)
*Animation focus: two groups randomised from one population, the difference in means settling as users arrive, a sequential test crossing a boundary, and a confounder shifting an apparent effect.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| ab-tests | A/B tests: randomisation and the difference in means | random assignment; treatment versus control; the estimator of the difference; standard error of a difference | product experiments, model rollouts |
| ab-analysis | Analysing an A/B test end to end | the two-proportion z-test; confidence interval for the lift; reporting a result with its uncertainty | conversion testing |
| peeking | Peeking, stopping early and sequential testing | the inflated false-positive rate under repeated looks; alpha spending; sequential boundaries (preview) | continuous monitoring |
| multiple-testing | Multiple comparisons and the false discovery rate | family-wise error; Bonferroni; Benjamini–Hochberg; many metrics, many segments | metric dashboards with hundreds of cuts |
| confounding | Confounding and why correlation is not causation | a confounder that opens a spurious path; adjusting by stratification; directed graphs (DAG) as a picture | observational studies |
| quasi-experiments | Quasi-experiments: difference-in-differences and natural experiments | the parallel-trends assumption; a before/after and treated/untreated design; interrupted time series | policy evaluation, rollouts without randomisation |

## Module 6 · Feature engineering & data pipelines  (`ds-features.html`)
*Animation focus: a raw record turning into a feature vector one transform at a time; a time-window feature sliding over an event log; leakage shown as a future value creeping into the past.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| encoding-categories | Encoding categorical variables | one-hot encoding; ordinal encoding; target encoding and its leakage risk; handling unseen categories | every tabular model |
| scaling | Scaling and transforming numeric features | standardisation; min–max scaling; log transforms; fitting the scaler only on training data | linear models, neural nets, k-NN |
| time-features | Date and time features | cyclical encoding of hour and weekday; lags and rolling windows; holidays | demand forecasting, churn |
| aggregates | Aggregate and window features | counts and means over windows; recency and frequency; keeping windows strictly in the past | recommendation, fraud detection |
| leakage | Data leakage: the silent killer | a feature that uses the future or the label; train/test contamination; leakage through preprocessing | any model that looks too good |
| pipelines | Reproducible pipelines and feature stores | fit/transform separation; a pipeline object; the same features at training and serving time; feature stores (preview) | production ML systems |

## Module 7 · Predictive modelling in practice  (`ds-modelling.html`)
*Animation focus: a train/validation/test split reshuffling; k-fold cross-validation rotating the held-out fold; a learning curve filling in as training size grows.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| split-validate | Train, validation and test splits | holdout sets; why the test set is used once; stratified splits; time-based splits for time data | every supervised project |
| cross-validation | Cross-validation and honest error estimates | k-fold; leave-one-out; variance of the estimate; nested CV when tuning | model selection |
| baselines | Baselines before complex models | majority-class and mean predictors; a simple linear model as the bar to beat; the cost of skipping baselines | sanity checks |
| metrics-choice | Choosing metrics: accuracy, precision, recall, RMSE and MAE | the confusion matrix; precision–recall trade-off; metrics that match the cost of errors; class imbalance | evaluation of every classifier |
| tuning | Hyperparameter tuning without fooling yourself | grid and random search; the validation set as a tuning set; overfitting to the validation set | model tuning |
| model-explanation | Explaining models: feature importance and partial dependence | permutation importance; partial dependence plots; the limits of importance as causation | regulated decisions, debugging |

## Module 8 · Time series & forecasting  (`ds-timeseries.html`)
*Animation focus: a series decomposed into trend, season and residual; a moving average smoothing the noise; a forecast fan widening with the horizon; a rolling-origin backtest moving forward.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| ts-structure | Time series structure: trend, seasonality and noise | additive versus multiplicative decomposition; stationarity intuition; autocorrelation at lags | demand, energy, traffic |
| smoothing | Moving averages and exponential smoothing | simple exponential smoothing with α; Holt's trend; the level and trend update equations | forecasting baselines |
| arima-intuition | ARIMA intuition: autoregression and differencing | AR(p) and differencing for trends; reading the autocorrelation plot; the intuition of ARIMA(p,d,q) | classic forecasting |
| backtesting | Backtesting: rolling-origin evaluation | train on the past, forecast the future; never shuffle time; rolling origin; the forecast error by horizon | every forecast |
| forecast-uncertainty | Forecast intervals and probabilistic forecasts | prediction intervals; quantile forecasts; the fan chart; calibration of intervals | inventory, capacity planning |
| ml-forecasting | Machine-learning forecasts and their pitfalls | lag features turning a series into a table; leakage in lags; global models across many series (preview) | retail demand, sensor forecasting |

## Module 9 · Communicating, deploying & responsible data science  (`ds-production.html`)
*Animation focus: a model's accuracy degrading as the input distribution drifts; a fairness metric shifting between groups; a monitoring dashboard raising an alert when a threshold is crossed.*
| id | Lesson | Must understand | AI link |
|---|---|---|---|
| reports-reproduce | Reports people can trust and reproduce | writing the method, the data and the limits; version control for analysis; re-running from raw data | research, audits |
| model-cards | Documenting datasets and models: datasheets and model cards | intended use; training data; known failure modes; evaluation by subgroup | responsible AI documentation |
| drift-monitoring | Monitoring: data drift, concept drift and alerts | population stability index; comparing live and training distributions; alert thresholds; retraining triggers | production ML |
| fairness | Fairness metrics and their trade-offs | demographic parity; equal opportunity; calibration; the impossibility of satisfying all at once | credit, hiring, health |
| privacy | Privacy: anonymisation, consent and aggregation | direct versus quasi-identifiers; k-anonymity; aggregation thresholds; why anonymisation often fails | personal data pipelines |
| career-path | Working as a data scientist: scope, collaboration and next steps | scoping work with stakeholders; the data-science role families; a portfolio of reproducible projects | career planning |
