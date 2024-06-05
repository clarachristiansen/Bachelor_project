# Bachelor Project: Prediction of train punctuality using ML
![ ](/Images/DSB_train.jpg)

This project, conducted in collaboration with the Danish train operating company DSB and the Technical University of Denmark, aimed to predict train punctuality on a daily basis for regional trains. Machine learning models such as ARIMA, GBDT (XGBoost & CatBoost), and Hybrid models achieved similar results, with an average prediction precision of around 9 percentage points, with the Hybrid model showing the best performance. Yet the precision was insufficient for practical application at DSB. Efforts such as model tuning and enhancing data quality were proposed to improve the precision. However, the inherent difficulty of the problem was likewise acknowledged and later highlighted through comparisons with state-of-the-art models. The feature importance analysis provided valuable insights by identifying key features which spanned across three categories: Timetable planning, employee attendance, and seasonality. Contrary, uncertainty quantification offered limited practical value due to its weak correlation with actual prediction errors. 

This initial exploration provides a comprehensive overview of the available data, potential models, and considerations regarding feature importance and uncertainty. By laying the groundwork for future research, this project aims to ultimately contribute to more efficient train operation and planning for DSB. Achieving these goals will require further resources and development.

**The structure of the repository is as follows:** 
- Environments
  - requirements.txt holds the needed packages and their version for all code except LSTM code
  - requirements_LSTM.txt holds holds the needed packages and their version for all code including LSTM code (contains Tensorflow)
- Data
  - Feature_selection.ipynb: Feature selection through Random Forest and filter methods. Initial data exploration (kinda messy) resulting in Feature information document from Appendix 10.1.3.
  - Feature_exploration.ipynb: Plotting of each feature. Examination of correlations. Forecasting of features with ARIMA, prediction horizon=1.
  - DMI_API.ipynb: Code for testing and using DMI API. DMI_API_HPC.ipynb holds the final code and was deployed on DTU's High Performance Computing (HPC) services.
  - DMI_data_merged.ipynb: Merges dataset with DMI data pulled via API.
  - Data_cleaning.ipynb: Cleaning of data and made two datasets: Orginal and Simple.
 
- Models
  - ARIMA_suitable.ipynb: Tests to see if ARIMA is suitable for data. Only for one route-station pair.
  - Time_series_models.ipynb: Understand time series models and their code.
  - ARIMA_strækning_station: Best model parameters for each route-station found via auto_arima. (HPC version also)
  - Individual_ARIMA_Boosted_CV_HPC.ipynb: Almost version control of all HPC code for cross-validation both for ARIMA, GBDT and Hybrid models. Coded to be called for each route-station i.e. much faster.
  - LSTM.ipynb: LSTM code with HPC version for cross-validation.

- Uncertainty Quantification
  - Uncertainty_quantification.ipynb: Uncertainty estimates based on XGBoost on route 20, station 19 with varying prediction horizon. Also have results from HPC version. Similarly to Individual_ARIMA_Boosted_CV_HPC.ipynb regarding version control.

- Visualization
  - Visualize_CV_shifted.ipynb: Visualization of all results from cross-validation: Error, CV score and features importance.
 
"Not used folder" is a folder for the files that are not used anymore but I could not delete. 
