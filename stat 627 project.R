library(dplyr)
#library(readxl) # using readxl::read_xlsx() with the original Excel file was 
# not working for me, so I converted the sheet with the data to a CSV file and 
# used readr::read_csv()
library(readr)

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
  ) |>
  filter(LowIncomeTracts == 1) # only looking at low-income tracts

# NOTE: If cells are NA, then the tract is not wide/long enough to be low-access
# at that threshold. Therefore, the tract technically has 0 low-access people 
# at that threshold
data_regression <- replace(data, is.na(data), 0) 