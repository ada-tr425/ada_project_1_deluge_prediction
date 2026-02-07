# References

## Websites
1) Stack Exchange. n.d. *stackoverflow* https://stackoverflow.com/questions [accessed 17-21 November 2025]
2) NumPy. n.d. *NumPy Documentation* https://numpy.org/doc/ [accessed 17-21 November 2025]
3) Python Software Foundation. n.d. *Python Documentation* https://docs.python.org/3/index.html [accessed 17-21 November 2025]
4) The Matplotlib Development Team. n.d. *Matplotlib Documentation* https://matplotlib.org/stable/ [accessed 17-21 November 2025]
5) Scikit Development Team. n.d. *Scikit Learn Documentation* https://scikit-learn.org/stable/ [accessed 17-21 November 2025]
6) XGBoost Development Team. n.d. *XGBoost Documentation* https://xgboost.readthedocs.io/en/stable/ [accessed 17-21 November 2025]
7) Google Gemini. Google Gemini response to Hassan Almarzooq. 17-21 November 2025. 
8) Folium Documentation [folium documentation](https://python-visualization.github.io/folium/latest/) [accessed 17-21 November]
9) ChatGPT. ChatGPT response to Kinen Kao. 17-21 November 2025.
10) Geopandas (https://geopandas.org/en/stable/docs/user_guide/mapping.html) [accessed 17-21 November]

## Books
None
## Journal Articles
None

## AI usage
I (Hassan Almarzooq) used Gemini (version: Gemini 2.5 Pro, publisher: Google, URL: https://gemini.google.com/app) to provide an example of how to override the default XGBoast fit method with log1p function for target variable, and to find out if there is a way to encode postcodes. I did not use it to generate any part of my submission directly.

- Google Gemini (2025). how to modify a class inherited of XGBoost.fit so that all of the y (target variable) applies a log1p transformation on y_train, y_val, y_test or any target. Gemini response to Hassan Almarzooq. Available at: https://gemini.google.com/share/a099eccb118d (Accessed: 18 November 2025).
- Google Gemini (2025). give me ideas on how to encode uk postcodes (categorical features) for a simple linear regression machine learning model. Gemini response to Hassan Almarzooq. Available at: https://gemini.google.com/share/2d1b051d2f32(Accessed: 17 November 2025).

- MS Copilot: Used it to help with adjusting the colormaps for folium plot, handling categorical and continuous data.
- MS Copilot: Used it to plot different layers of data points (rainfall and level data) on folium map.
- MS Copilot: Used it to help with adjusting and adding legends for folium map.
- MS Copilot: Used it to help with adding caption inside folium map.
- ChatGPT (2025). How to safely remove old files that I don't need on Github. ChatGPT response to Kinen Kao. Available at: https://chatgpt.com/s/t_691f278957308191a95b7149e03d358b
- ChatGPT (2025). How to impute missing values in an unlabelled dataframe using values learned from a labelled dataset. ChatGPT response to Kinen Kao. Available at: https://chatgpt.com/s/t_691f22a3ccf08191ab92d9ccc62f1d7a
- ChatGPT (2025). How to find if there are any non-numeric items inside a dataframe column that is supposed to be all numeric values. ChatGPT response to Kinen Kao. Available at: https://chatgpt.com/s/t_691f24743d9c819181b9a02e2cfce0eb

I (Tingyu Rao) used Chat GPT (version: GPT 4.0, publisher: DeepLearnig, URL: https://chatgpt.com) to seek guidance on row-wise data normalization, calling other interfaces within the same class, and utilizing existing data in Python. I did not use it to generate any part of my submission directly.

- Chat GPT. How to implement row-wise normalization (including Min-Max and Z-score) for numerical features in district-level pandas DataFrames, handle missing values before normalization, and apply efficient techniques to avoid performance issues with large district datasets. Available at: https://chatgpt.com/share/691f3585-cc78-8007-8714-34036891bfc6 (Accessed: 19 November 2025).
- Chat GPT. How to call sibling methods within the same Python class, pass parameters and handle return values between them, maintain data consistency, and prevent circular dependencies for effective interface design. Available at: https://chatgpt.com/share/691f35ea-f37c-8007-9297-0d944dc8aad5 (Accessed: 19 November 2025).
- Chat GPT. How to efficiently reuse data loaded in a Python class’s init method across multiple interfaces, maintain data consistency, cache data to avoid redundant loading, and merge preloaded existing data with new input data for interface-specific processing. Available at: https://chatgpt.com/share/691f367b-be6c-8007-8235-4191594507c8 (Accessed: 18 November 2025).

I (Zhongkai Yuan) used ChatGPT 5.1 model (Publisher: OpenAI) to explain things, help fix issues, and adjust formatting.
- ChatGPT was used to help diagnose repeated ModuleNotFoundError: No module named 'flood_tool' issues during pytest collection.
- ChatGPT provided conceptual clarification on: where training functions should be placed inside fit_to_data(), how predict_historic_flooding() should call the model, and how to correctly pass preprocessed postcode features into the RandomForest model.
- ChatGPT explained typical usage of: joblib.dump(), Path(__file__).resolve() and writing CV reports into the /reports directory. This guidance helped me understand the file-saving structure.
- ChatGPT helped me structure several PowerPoint slides for the presentation
-ChatGPT (OpenAI) answered questions about: overfitting checks, decision thresholds, and why training inside the Tool.py uses the preprocessed postcode dataset.

I (Chuan Ju) used Gemini3 model to explain things, help fix issues, and adjust formatting.
- Gemini helped me analyze the variable domain problem:
scoring/test_scorable.py::test_predict_local_authority - UnboundLocalError: cannot access local variable 'idx' where it is not associated with a value
- Gemini helped me diagnose that the error in the scorer's main function was caused by a data type error in the index:IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
- Gemini diagnosed that my model might be overfitting, so I followed its advice and optimized the hyperparameters to avoid this problem.
- Gemini gave me ideas on how to visualize the output: told me how I could create a heatmap of risk and a boundary graph of local authority.
- Gemini helped me organize the viewpoints of the model to complete the PPT and report.

I (Kaiqing Deng) used AI tools during the development of the Seven-Class Flood Risk Model purely for debugging support, conceptual clarification, and code-refactoring guidance. AI did not use it to generate any part of my submission directly.
- ChatGPT (2025):
Help to integrate my model `seven_class_tool.py` link to `tool.py`
- Microsoft Copilot: 
Consulted briefly to understand different strategies for managing imputation logic inside machine-learning workflows.
- Used AI to clarify edge-case conditions that cause model or visualisation failures.
- Used AI to validate data-cleaning logic and ensure correct handling of missing values.

I (Indah Mustika Dewi) Used ChatGPT 5.1 to visualise map for Historical model Available at: (https://chatgpt.com/share/69201caa-4da4-8001-82fc-08a311d30307)
- Chat Gpt was used to explain how to plot data using folium
- Chat Gpt was used to explain geopandas and merge 2 CSV files into 1 files.


