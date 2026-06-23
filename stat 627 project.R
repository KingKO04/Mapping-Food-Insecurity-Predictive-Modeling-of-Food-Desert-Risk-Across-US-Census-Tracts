## STAT-627 Final Project
## Names: Kaodi Onwumechili and Inaya Ahmed 

## Mapping Food Insecurity: Predictive Modeling of Food Desert Risk Across US Census Tracts

#library(readxl) # using readxl::read_xlsx() with the original Excel file was 
# not working for me, so I converted the sheet with the data to a CSV file and 
# used read.csv()
library(relaimpo) # relative importance for the linear (forward stepwise) model
library(pls)     # for principal components regression (pcr)
library(corrplot)
library(caret)   # for train/test split and cross-validation utilities
library(glmnet)
library(tidyverse)
library(car)
library(vip)

## DATA CLEANING

#data <- read_xlsx("FoodAccessResearchAtlasData2019.xlsx", 3)

data <- read.csv("Food Access Research Atlas Data 2019.csv") # load data in

row.names(data) <- data$CensusTract # ID column

data <- select(data, -CensusTract) # remove duplicate

data <- replace(data, data == "NULL", NA) # change NULL cells to NA

# variables that need to be factors
flags <- c(
  "Urban", 
  "GroupQuartersFlag", 
  "LILATracts_1And10",
  "LILATracts_halfAnd10",
  "LILATracts_1And20",
  "LILATracts_Vehicle",
  "HUNVFlag",
  "LowIncomeTracts",
  "LA1and10",
  "LAhalfand10",
  "LA1and20",
  "LATracts_half",
  "LATracts1",
  "LATracts10",
  "LATracts20",
  "LATractsVehicle_20"
)

# variables that need to be numeric
numbers <- colnames(data)[c(4, 5, 7, 8, 15, 16, 25:length(data))]

data <- data |>
  mutate(
    across(all_of(flags), as.factor), # factorizing flags
    across(all_of(numbers), as.numeric), # numerizing numbers
    # low access share. if urban, >0.5 mi. if rural, >10 mi.
    lashare = ifelse(Urban == 1, lapophalf / Pop2010, lapop10 / Pop2010)
  )

# Only looking at low-income tracts for regression
data_li <- filter(data, LowIncomeTracts == 1)

# NOTE: If cells are NA, then the tract is not wide/long enough to be low-access
# at that threshold. Therefore, the tract technically has 0 low-access people 
# at that threshold
data_regression <- replace(data_li, is.na(data_li), 0) 

## EDA 
# --- Summary statistics for the regression target and key predictors ---
summary(data_regression$lashare)
summary(data_regression$MedianFamilyIncome)
summary(data_regression$PovertyRate)
summary(data_regression$TractHUNV) # housing units with no vehicle

# --- Distribution of the regression response ---
ggplot(data_regression, aes(x = lashare)) +
  geom_histogram(bins = 40, fill = "steelblue", color = "white") +
  labs(
    title = "Distribution of Low-Access Share Among Low-Income Tracts",
    x = "Share of Tract Population Lacking Supermarket Access",
    y = "Count of Tracts"
  ) +
  theme_minimal()

# --- Urban vs. rural comparison of low-access share ---
ggplot(data_regression, aes(x = Urban, y = lashare, fill = Urban)) +
  geom_boxplot() +
  labs(
    title = "Low-Access Share by Urban/Rural Status",
    x = "Urban (1) vs. Rural (0)",
    y = "Low-Access Share"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# --- Correlation among candidate numeric predictors ---
# Selected to mirror the socioeconomic/geographic predictors named in the
# planned approach (income, poverty, vehicle access, population density)
eda_vars <- data_regression |>
  select(MedianFamilyIncome, PovertyRate, TractHUNV, Pop2010, lashare) |>
  mutate(across(everything(), as.numeric))

corr_matrix <- cor(eda_vars, use = "complete.obs")
corrplot(corr_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.7,
         title = "Correlation Among Candidate Predictors",
         mar = c(0, 0, 1, 0))

# --- Class balance check for the classification target ---
table(data_regression$LILATracts_1And10)
prop.table(table(data_regression$LILATracts_1And10))


## REGRESSION MODELING 
## Forward stepwise regression vs. principal components regression
## Response: lashare (proportion of low-income tract lacking supermarket access)

set.seed(627)

# --- Train/test split ---
train_idx <- createDataPartition(data_regression$lashare, p = 0.8, list = FALSE)
train_set <- data_regression[train_idx, ]
test_set  <- data_regression[-train_idx, ]

# Candidate explanatory variables: socioeconomic and geographic features
# consistent with the planned approach in Section 5c
reg_predictors <- c(
  "MedianFamilyIncome", "PovertyRate", "TractHUNV", "Pop2010",
  "TractSNAP", "TractSeniors", "TractKids", "TractWhite", "TractBlack",
  "TractAsian", "TractHispanic", "Urban"
)

reg_formula <- as.formula(
  paste("lashare ~", paste(reg_predictors, collapse = " + "))
)

# --- Full model (upper scope for stepwise search) ---
full_model <- lm(reg_formula, data = train_set)

# --- Null model (lower scope for stepwise search) ---
null_model <- lm(lashare ~ 1, data = train_set)

# --- Forward stepwise regression (AIC-based selection) ---
forward_model <- step(
  null_model,
  scope = list(lower = null_model, upper = full_model),
  direction = "forward",
  trace = 0
)
summary(forward_model)

# Predicted values and R-squared on the held-out test set
forward_pred <- predict(forward_model, newdata = test_set)
forward_r2 <- cor(forward_pred, test_set$lashare)^2
forward_rmse <- sqrt(mean((forward_pred - test_set$lashare)^2))

# --- Principal Components Regression ---
pcr_model <- pcr(
  reg_formula,
  data = train_set,
  scale = TRUE,
  validation = "CV"
)

# Select number of components minimizing CV error
validationplot(pcr_model, val.type = "MSEP")
ncomp_best <- which.min(RMSEP(pcr_model)$val[1, 1, ]) - 1

pcr_pred <- predict(pcr_model, newdata = test_set, ncomp = ncomp_best)
pcr_r2 <- cor(as.vector(pcr_pred), test_set$lashare)^2
pcr_rmse <- sqrt(mean((as.vector(pcr_pred) - test_set$lashare)^2))

# --- Model comparison: percentage of variance explained ---
model_comparison <- data.frame(
  Model = c("Forward Stepwise", "PCR"),
  R_squared = c(forward_r2, pcr_r2),
  RMSE = c(forward_rmse, pcr_rmse),
  N_Predictors = c(length(coef(forward_model)) - 1, ncomp_best[[1]])
)
print(model_comparison)


## VARIABLE IMPORTANCE/INTERPRETABILITY ANALYSIS
# --- Relative importance of predictors retained in the forward model ---
# lmg metric: decomposes R^2 by averaging over orderings (Lindeman, Merenda, Gold)
relimp <- calc.relimp(forward_model, type = "lmg", rela = TRUE)
print(relimp)
plot(relimp)

# --- Standardized coefficients for direct magnitude comparison ---
std_coefs <- data.frame(
  Predictor = names(coef(forward_model))[-1],
  Coefficient = coef(forward_model)[-1]
)
std_coefs <- std_coefs |> arrange(desc(abs(Coefficient)))
print(std_coefs)

ggplot(std_coefs, aes(x = reorder(Predictor, abs(Coefficient)), y = Coefficient)) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(
    title = "Forward Stepwise Regression: Predictor Coefficients",
    x = "Predictor",
    y = "Coefficient (Effect on Low-Access Share)"
  ) +
  theme_minimal()

# --- Loadings plot for PCR (which original variables drive top components) ---
loadingplot(pcr_model, comps = 1:2, legendpos = "topleft",
            main = "PCR Loadings: First Two Principal Components")

## Classification modeling
# LASSO
# share of occupied housing units with vehicles
data_classification <- data |>
  mutate(vehicle_share = 1 - (TractHUNV / OHU2010)) |>
  select(LILATracts_halfAnd10, PovertyRate, vehicle_share, Urban) |>
  drop_na()

set.seed(627)
train <- sample(nrow(data_classification), .8 * nrow(data_classification))
data_train <- data_classification[train, ]

# poverty rate -> PovertyRate
# vehicle availability -> vehicle_share
# population density -> Urban
# distance to nearest supermarket -> ???
train_mat <- model.matrix(glm(
  LILATracts_halfAnd10 ~ PovertyRate + vehicle_share + Urban, 
  "binomial", 
  data_train
))
data_test <- data_classification[-train, ]
test_mat <- model.matrix(glm(
  LILATracts_halfAnd10 ~ PovertyRate + vehicle_share + Urban, 
  "binomial",
  data_test
))

set.seed(627)
data_lasso <- cv.glmnet(
  train_mat, 
  data_train$LILATracts_halfAnd10,
  family = "binomial",
  type.measure = "auc"
)

best_t <- -1
best_conf_mat <- NULL
t_pos_rate <- c()
f_pos_rate <- c()
for (t in seq(0, 1, .01)) {
  conf_mat <- confusionMatrix(
    as.factor(
      ifelse(predict(data_lasso, newx = test_mat, type = "response") > t, 1, 0)
    ),
    data_test$LILATracts_halfAnd10,
    "1"
  )
  best_bal_acc <- ifelse(
    is.null(best_conf_mat), 
    -1, 
    best_conf_mat$byClass[["Balanced Accuracy"]]
  )
  t_pos_rate <- c(t_pos_rate, conf_mat$byClass[["Sensitivity"]])
  f_pos_rate <- c(f_pos_rate, 1 - conf_mat$byClass[["Specificity"]])
  # prioritize identifying LILA areas
  if (conf_mat$byClass[["Sensitivity"]] > .95)
    # do not want to overpredict LILA areas; main measure for comparison
    if (best_bal_acc < conf_mat$byClass[["Balanced Accuracy"]]) {
      best_conf_mat <- conf_mat
      best_t <- t
    }
}

# Logit - WINNER
data_logit <- glm(
  LILATracts_halfAnd10 ~ PovertyRate + vehicle_share + Urban, 
  "binomial", 
  data_train
)
vif(data_logit) # multicollinearity check

best_th <- -1
best_cm <- NULL
tpr <- c()
fpr <- c()
for (th in seq(0, 1, .01)) {
  cm <- confusionMatrix(
    as.factor(ifelse(predict(data_logit, data_test, "r") > th, 1, 0)),
    data_test$LILATracts_halfAnd10,
    "1"
  )
  best_ba <- ifelse(
    is.null(best_cm),
    -1, 
    best_cm$byClass[["Balanced Accuracy"]]
  )
  tpr <- c(tpr, cm$byClass[["Sensitivity"]])
  fpr <- c(fpr, 1 - cm$byClass[["Specificity"]])
  if (cm$byClass[["Sensitivity"]] > .95)
    if (best_ba < cm$byClass[["Balanced Accuracy"]]) {
      best_cm <- cm
      best_th <- th
    }
}

plot(f_pos_rate, t_pos_rate, main = "Logit ROC Curve", col = "blue") # ROC Curve
abline(0, 1, col = "red")
best_conf_mat # confusion matrix
vip(data_lasso, mapping = aes()) # importance
